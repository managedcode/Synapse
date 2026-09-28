# ADR-007: concurrent sessions and continuous batching in one model instance

Status: Accepted. Date: 2026-09-28.

## Context

A loaded `Qwen2Model` owned one scratch area and one KV cache, so concurrent
`Generate` calls on the same instance would corrupt each other. The owner
asked for many parallel requests, for example five at once. On CPU, a decode
step is bound by reading every weight row, as ADR-006 measured. The kernels
already evaluate several tokens per weight-row pass, so tokens from different
requests can share one pass. This is the local form of `elastic-inference.plan.md`
E7 (TASK-GEN-007).

## Decision

- **Async API.** `ITextGenerationModel.GenerateAsync(promptTokens,
  maximumNewTokens, cancellationToken)` is thread-safe. Each call becomes a
  session on a scheduler that the model owns. The synchronous `Generate`
  stays. It uses a dedicated direct session and serializes with scheduler
  steps through one execution gate. Nothing calls async code synchronously.
- **KV ownership.** Each session owns one KV slot for its whole life. Slots
  are FP32 `[layer][position][kvWidth]`, allocated lazily and bounded by
  `ModelLoadOptions.MaximumConcurrentSessions` (default 4). The scheduler
  thread is the only writer of a scheduled slot. Positions always start at
  zero, and attention reads only positions `0..p` of its own slot, so a reused
  slot never exposes earlier tokens.
- **Step composition.** Each step feeds, in order:
  1. every session in decode, with exactly one token each (decode comes
     first);
  2. prompt tokens of sessions still in prefill, first-in first-out, up to
     `PrefillChunkTokens` tokens in total.

  One forward pass processes the whole step. Each prompt-completing or decode
  token gets its own logits row. The vocabulary projection reads its weights
  once for all those rows.
- **Exactness.** Every per-token operation is the same one used by a
  single-session run: row dots, RoPE, KV write, attention over the owning
  slot, and the logits row. A batched run therefore produces the same tokens
  and bitwise logits as independent runs.
- **Admission.** When every slot is busy, new requests wait first-in
  first-out. Nothing is silently degraded or truncated.
- **Cancellation.** A canceled request completes as canceled at the next step
  boundary. Its slot is released and later reused.
- **Reference backend.** `reference` runs one request at a time under the same
  gate. It does not batch.

## Consequences

- Memory is the shared mapped weights plus active slots × KV bytes. At a
  512-token context that is 12.6 MB per Qwen2.5-0.5B slot.
- Throughput with N decoding sessions rises while one weight pass serves all N
  tokens. Per-session latency rises with step size. The CLI can measure both
  with `--concurrent-requests`.
- Tests: `RaggedBatchMatchesIndependent`, `LongPrefillDoesNotStarveDecode`,
  `CancelledSlotReusedSafely`, plus KV parity tests
  (`IncrementalDecodeMatchesFullPrefill`, `GenerationRepeatsAfterLongerPrompt`,
  `LongPromptAcrossChunksMatchesSequential`, context bounds).
- Distributed mode keeps this worker-local loop. A future `SessionGrain`
  records only which worker incarnation and epoch own a session. Batches,
  tokens, activations, and KV never cross Orleans (Architecture, INV-012).
