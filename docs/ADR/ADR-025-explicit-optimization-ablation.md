# ADR-025: explicit optimization controls and long-context ablation

Status: Accepted for bounded local measurements. Date: 2026-10-03.
Requirement: `REQ-CTX-009`. Related memory decision: ADR-017 / `REQ-CTX-010`.
Broader runtime and benchmark tasks remain in progress.

## Decision

Preparation and execution remain separate. Convert a supported source to
`.synapse` explicitly before runtime. The lossless prepared weights are used
for all quality-preserving memory/performance comparisons. Weight averaging,
its codecs, conversion flags and runtime gates are removed at the owner's
request after real retrieval failures; ADR-023 retains the rejected decision.

`generate` and `score` accept `--optimization off|dense|custom` (default
custom preserves existing behavior). `off` selects Reference, FP32 KV, full
layers and dense attention without speculation. `dense` keeps the requested
backend with FP32 KV, full layers and dense attention. Contradictory settings,
unknown/duplicate flags and a draft with concurrent generation fail explicitly.
`custom` exposes supported backend, KV, page, layer and speculation settings.
JSON records requested mode and effective runtime profile.

Dense means full execution, not bitwise equality across different backends:
CPU activation quantization and GPU accumulation remain explicit. Prefix reuse
is measured on real repeated requests. FP16 KV is an explicit precision choice;
quality and speed must be checked together. Page activation and layer drops
stay opt-in approximations and cannot become defaults from timing alone.

The existing locked `Qwen2Model.AllocatedKvBytes` diagnostic becomes public
to observe allocator memory directly. ADR-017 defines growth, request-bound
contraction, batching ownership and disposal. Single-request CLI generation
bounds executor sessions/logits rows to its actual needs; speculation reserves
its verification rows and scoring/concurrent commands retain their own bounds.

## Measurement boundary

`optimization-eval --plan <json> --output <new.json>` runs fresh real
`optimization-worker` child processes against prepared packages only. No
compiler is called. Profiles have a unique dense baseline on the same backend,
with original weights, FP32 KV, all layers, dense attention and reuse disabled.

The strict bounded plan records model/haystack paths, context/prompt limits,
depths, tasks, seed, output cap, scored tail, warmups, measurements, threads,
optional RoPE and repeated-prompt policy. Profiles record optional package,
backend, KV precision, reuse, page activation and explicitly evidenced drops.
Unsupported profiles fail before measurement.

Needle, multi-key/distractor and variable-tracking cases use identical original
token IDs for every profile. Original source provenance and actual prompt
tokenizer compatibility are checked. This is prompt compatibility, not a
digest of the entire vocabulary. Profile order rotates by case/round. Cold and
repeated requests are separate; repeat without reuse controls prefix speed.

Each child loads one model, generates, optionally repeats and teacher-forced
scores a bounded prompt suffix. It records actual package/source/file/graph,
runtime/kernel and token identities, requested/effective controls, input/output
counts, load, TTFT, prefill/generation/decode time, allocated KV, observed
process memory, output IDs/text/answer grade, per-position NLL and greedy IDs.
CPU page trials score one row per step to exercise the approximate path.
Aligned comparisons report greedy agreement, absolute NLL drift and tail
perplexity ratio. Tail perplexity on retrieval prompts diagnoses numerical
drift; it does not qualify held-out language quality.

Evidence retains finite mean NLL (log-perplexity) even when its exponential
cannot be represented as a finite positive double. In that case perplexity is
null and an explicit range status is recorded; JSON never publishes Infinity
or NaN. Ratios use the difference of log-perplexities and record their own
range status, so two overflowing perplexities can still have a finite ratio.
Non-finite scoring values fail explicitly while retaining earlier generation
and memory phases. Each candidate must resolve to the same backend enum as the
dense baseline; equivalent spelling aliases do not create another backend.

Cancellation/timeouts terminate the real child tree, await exit and retain
completed evidence with incomplete status. Missing hardware is
`not_run_missing_hardware`. Unsupported combinations fail explicitly. Atomic
publication never overwrites a caller's output. There is no fake subject,
automatic winner or quality promotion. A speed claim requires paired raw
results with corresponding quality, output-length and host-load observations.

## Acceptance mapping

- `AC/TEST-CTX-009-1`: off/dense/custom, strict flags and contradictions.
- `AC/TEST-CTX-009-2`: strict bounded plans, unique dense baseline and
  prepared-only execution.
- `AC/TEST-CTX-009-3`: real generation/scoring children, aligned IDs,
  quality metrics and package/source/file/graph provenance.
- `AC/TEST-CTX-009-4`: controlled repeats and actual prefix reuse.
- `AC/TEST-CTX-009-5`: rotated pairing, cancellation/failure retention and
  no implicit conversion or automatic quality verdict.

General graph scheduling, adaptive weight precision, GPU page approximation
and broad model/held-out qualification remain open.
