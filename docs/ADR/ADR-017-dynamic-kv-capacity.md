# ADR-017: Dynamic KV capacity (memory follows the context in use)

Status: Accepted. Date: 2026-09-28. Amends ADR-012 (KV slots) and ADR-013 (context limits).

## Context

The owner asked for practical large contexts beyond 16k and 32k, and for
"dynamic context". Until now a KV slot was sized for the whole configured
context the first time it was used.

- **Metal.** 12 KiB per position (FP16) or 24 KiB (FP32) times `ContextSize`.
  A 131,072-token instance holds 1.5–3 GiB of KV for a 100-token prompt.
- **CPU.** FP32 arrays of `ContextSize × kv width` per layer.

The consequence was that callers had to choose between memory and headroom,
and a large default context was too expensive to offer. llama.cpp also
preallocates `-c` tokens, while MLX grows its cache in steps.

## Decision

- **A slot has a capacity.** `ContextSize` stays the hard limit: positions at
  or above it fail, and YaRN still decides what is allowed (ADR-013). A
  slot's allocated capacity starts small and grows on demand.
- **Growth policy.** Capacity is a multiple of 1,024 positions (configurable
  internally for tests). It starts at the first need rounded up. It grows to
  `min(ContextSize, max(needed, 2 × current))`. Doubling keeps the total
  copy cost linear in the tokens seen.
- **Metal layout and copy.**
  - The slot layout stays `[layer][KV head][position][64]` for K, then the
    same for V. The position stride is the slot's own capacity.
  - The slot table carries each slot's capacity next to its address, and
    every kernel that addresses KV (`rope_kv`, prompt and decode attention)
    uses that stride instead of the instance context.
  - Growth allocates a zero-filled buffer. It copies each (layer, head)
    region of K and V with a blit (`layers × kv_heads × 2` copies) in a
    command buffer that completes before the step is encoded.
  - The padding of one attention tile stays at the end of the slot.
- **CUDA.** Until it can be tested on hardware, CUDA keeps full-size slots
  (capacity = context) and passes the same table layout.
- **CPU.** `Qwen2KvCache` stores `[position][kv width]` per layer, so its
  layout does not depend on capacity. It grows by resizing with the same
  policy.
- **Reservation.** A request that knows its size (prompt plus maximum new
  tokens for `Generate`, the token count for scoring) sizes the direct slot
  once, rounded up to the growth unit, before it evaluates anything
  (`synapse_gpu_decoder_reserve`). Growth copies briefly hold the old and the
  new buffer: a 32k prompt peaked at 686 MiB (FP16 KV) while growing, and at
  534 MiB with a reservation.
- **Numerics.** Unchanged. Growth copies bytes, and kernels read the same
  values at the same positions. A test pins bitwise-equal logits between a
  slot that grows many times and a slot allocated at full size.
- **Observability.** The native ABI gains `synapse_gpu_decoder_kv_bytes`
  (ABI version 2), so tests and evidence can report allocated KV bytes. The
  descriptor gains `kv_growth_positions` and an explicit `pad`: 136 bytes.

## Consequences

### 2026-10-03 amendment: contraction and release (`REQ-CTX-010`)

Profiling and the long→short behavior test expose growing-only retention: a
completed long request leaves its full allocation resident for a later short
request. CPU executor disposal also retained arrays while the disposed model
object remained reachable. Both are allocator issues, independent of model math.

`Reserve` now computes the request rounded up to the same growth unit. It grows
as before, and contracts when that rounded need is at most one quarter of the
current capacity. Otherwise it keeps the allocation. This hysteresis avoids
reallocating for small request-size changes. The instance context remains the
hard limit; no active token or layer is discarded. The new request fits in the
new capacity, including every prefix position claimed for exact reuse.

CPU resizing preserves the surviving position bytes. Metal copies the smaller
of old/new position ranges per layer/head for K and V with their respective
strides and waits for the transfer before executing. Native CUDA retains its
existing full-context policy until real hardware qualification; no contraction
or missing-device test is claimed there.

Continuous batching reserves the prompt-plus-output bound of each newly
assigned slot under the model execution gate before its first step. An active
slot is never contracted by another request. Direct generation and scoring
already reserve under that same gate. Disposal joins the scheduler, takes the
gate and releases CPU arrays/slots and native resources; allocated KV is zero
even when the disposed managed model remains referenced.

Closing the model rejects new direct, asynchronous, tokenizer and digest work
before touching released storage. Asynchronous admission rechecks the closed
state under the execution gate, so it cannot enqueue into a joined scheduler.
Concurrent disposal is serialized and idempotent. Reentrant disposal from a
direct execution/progress callback fails with `InvalidOperationException`
before closing the owner: joining a scheduler from inside that gate could
deadlock, and releasing its active native mapping would invalidate the caller.
Scheduler-thread callbacks receive the same rejection before any state change,
because joining the current thread would deadlock. Reentrant direct generation,
scoring and internal direct math fail with `InvalidOperationException`; a nested
request cannot contract or overwrite the outer operation's active KV. Metadata,
tokenizer and allocation reads stay available to progress callbacks. Enqueuing
asynchronous work is allowed; its math executes after the active gate is released.

The existing `AllocatedKvBytes` diagnostic becomes public. This is allocator
memory, not process RSS or the total mapped model footprint. The local
`memory-eval` command consumes an already prepared package and records
load→short→long→short→dispose allocation, timings, same-backend cold numerical
comparators and real process observations. It never forces a GC to claim OS
release. `TASK-CTX-010` stays in progress until its focused/full gates and real
short→long→short measurements are recorded.

Acceptance: `AC/TEST-CTX-010-1` contraction/hysteresis preserves bytes;
`-2` direct cold numerical and prefix-reuse parity; `-3` batch slot contraction
and disposal; `-4` prepared-only phase measurement with truthful provenance,
quality, cancellation/failure handling and immutable output publication.

- A large `ContextSize` costs memory only when it is used, so the CLI and
  SDK can offer long contexts without a memory penalty on short prompts.
- A growth step costs one extra command buffer and a copy of the used KV,
  a few milliseconds at most for this model at 32k. It happens a logarithmic
  number of times per session.
- Tests: `GrowingKvSlotMatchesFullSlotBitwise`,
  `KvBytesFollowTheContextInUse`, `CpuKvCacheGrowsAndKeepsContents`,
  `GenerateReservesPromptAndOutputOnce`, and the Rust
  `kv_capacity_grows_by_unit_and_doubling_up_to_the_context`.
