# GPU kernels and long-context plan

Scope: Apple Metal and NVIDIA CUDA execution of brand-neutral dense decoders, and honest behavior at 32k,
40k, and 128k tokens. Decisions: ADR-012 (GPU backends) and ADR-013 (context limits and RoPE scaling).
Requirements and tests: `docs/Features/GpuKernels.md` and `docs/Features/LongContext.md`.

Design references were studied from source (license noted, nothing copied):

- llama.cpp ggml-metal and ggml-cuda (MIT): the Q8_0 matrix-vector lane mapping and the 64×32 GEMM tile;
- MLX (MIT): split-key decode attention;
- DwarfStar / antirez ds4: disk-resident KV for long contexts;
- vLLM (Apache-2.0): paged KV and chunked prefill.

## Steps

- [x] CTX.1 `TASK-CTX-001`: no silent context clamp. The red state was the compile failure of
  `ContextBeyondTrainedLengthFailsExplicitly`; it is green now.
- [x] CTX.2 `TASK-CTX-002`: explicit YaRN profile, one C# frequency source for every backend, and the
  profile in the Model IR fingerprint and the runtime profile. Green: `YarnFrequenciesMatchFp64Formula`,
  `UnscaledFrequenciesStayBitwise`, `YarnContextExtendsLimitExactly`, `ContradictingScalingFails`.
- [x] CTX.3a `TASK-CTX-003`: `--tokens-file`, `--rope-scaling yarn:<factor>:<trained>`, and prefill
  progress on stderr. `CliTokensFileMatchesInlineTokens` and `CliRejectsMalformedTokensFile` were red
  (usage exit), then green.
- [x] GPU.1 `TASK-GPU-001`: `synapse-gpu` crate and C ABI, device probe, explicit failures, and the
  `KernelBackend` rename. Rust validation and scheduling tests pass (10).
- [x] GPU.2 `TASK-GPU-002`: Metal dense decoder with the pinned continuation, reference tolerance,
  continuous batching, and the brand-neutral `DenseDecoderLayout` refactor. 7/7 Metal tests pass locally.
- [x] GPU.3 `TASK-GPU-003`: prompt-run GEMM, split decode attention, and a dedicated low-memory decode
  kernel.
  - Tests pass: `MetalPromptRunsTrackReference`, `MetalLongPromptAcrossChunksTracksManagedCpu`, and
    `MetalSplitDecodeTracksPromptRunAttention`.
  - The chunk regression test failed with the old null-logits bug restored and passes with the fix.
  - Watchdog-safe submissions were added after two 131k runs were killed.
- [x] CTX.3b: raw long-context evidence under `benchmarks/results/`.
  - Speed at 40k and 131k.
  - Pass-key fixture from pinned `llama-tokenize` segments. The 120k llama.cpp control was recorded as
    identical, but it lost the trailing newline (QUAL.3).
  - Native window: 9 of 9.
  - YaRN ×4 with FP16 KV: 3 of 3 at 65k, and 2 of 3 at 120k (the 10%-depth key was lost).
  - The llama.cpp control on the failed prompt is recorded in `docs/Features/LongContext.md`.
- [x] GPU.4 FP16 KV profile (`--kv-precision f16`) for Metal.
  - The tests failed first (red: the type did not compile), then passed: pinned continuation, 1% of the
    reference range, split decode, CPU rejection, and the CLI profile.
  - At 32k, decode rose from 75 to 93 tok/s.
  - It stays opt-in; FP32 remains the default.
- [ ] GPU.5 decode matrix-vector bandwidth.
  - Done: packed `char4` code loads, bitwise-equal. Short-context decode went from 135 to 164 tok/s.
  - Open: more rows per threadgroup, and GPU-friendly repacked weights from the ADR-004 package compiler
    (separate scales and codes, 16-byte alignment).
- [ ] GPU.10 prompt time (first token).
  - Done: prompt-run attention reads K/V straight from the slot, like decode. Each simdgroup owns a private
    16 KiB region with no threadgroup barrier; before, a staged tile took 26.5 KiB and allowed one threadgroup
    per core. All 17 Metal tests pass.
  - Measured: time to first token at 4k/8k/16k/32k went from 1.41/3.98/11.96/40.5 s to 1.34/3.63/10.28/32.6 s.
    llama.cpp takes 0.85/2.47/7.40/26.6 s.
  - Done (2026-09-29): scores and outputs stay in registers as each lane's fragment pair (`lane_pair`, MLX's
    layout), with no threadgroup round trip and no diagonal rescale. 32k time to first token is now 27.0 s and
    16k is 9.0 s. `SYNAPSE_GPU_PROFILE=1` reports GPU time per kernel.
  - Not taken: two eight-row halves per simdgroup share each K/V load but spilled (32k: 38.1 s).
  - Open: the prompt GEMM. At 512 tokens Synapse takes 179–185 ms against llama.cpp 108–126 ms, about
    2.2 TFLOPS. Three experiments did not close the gap:
    - half-precision weight tiles: -5%, rejected because the rounding changes the exact Q8_0 x FP32 profile;
    - contiguous 8x8 block layout: -3%;
    - 64x64 tiles: 3.4–4x slower.

    The pipelined, vectorized staging was kept (bitwise identical). Next: GPU counter profiling instead of
    guessing.
- [x] TOK.1 `TASK-TOK-001` (ADR-014): repo-owned byte-level BPE tokenizer read from the GGUF arrays,
  ChatML template, `synapse tokenize`/`detokenize`. The tests were red (types missing), then green.
  Corpus parity against `llama-tokenize --no-escape`: 702 of 702 runs identical (351 files, 540,679 tokens).
- [x] QUAL.1 `TASK-CTX-004` (ADR-015): teacher-forced `Score` on every backend, `ScoringRowsPerStep`,
  `synapse score` with the llama-perplexity protocol, and `generated_text` in `generate`. The tests were red
  (compile failure, and a missing JSON field for `generated_text`), then green. Perplexity parity runs are
  recorded in `docs/Features/LongContext.md`.
- [ ] QUAL.2 `TASK-CTX-005` (ADR-015): exact-answer task suite (needle, multi-key, variable tracking) across
  Synapse Metal FP32/FP16, llama.cpp Metal, and MLX (a separate-weights cohort, 32k and below). Harness
  tests are green; the recorded runs are in `docs/Features/LongContext.md`.
- [x] QUAL.3 corrected the 120k llama.cpp pass-key control. llama.cpp `-f` drops the prompt's trailing
  newline, so the first control evaluated 119,990 tokens. On the compensated file (119,991 identical
  tokens), llama.cpp answers `6`, exactly like Synapse. The evidence file records both runs.
- [ ] QUAL.4 `TASK-CTX-006` (ADR-016): query-aware KV page activation, the owner's "smart context" idea as
  partial activation over the context. Done: the C# reference selection on the managed and native CPU
  backends, and `--kv-pages` in `generate` and `score`. The tests were red (types missing), then 5 of 5
  green, including bitwise equality with dense for a covering budget. The page size (16/32/64) and
  `--first-scored` were added test-first. CPU decode-mode quality is recorded: key-bound selection beats
  the equal-budget random control at every budget; 16-token pages with 2,048 tokens cost +3.4% at 8k and
  +6.2% at 16k. Open: needle tasks under the profile, the Metal kernels matched to the C# selection (the
  only way to measure speed and memory), then a cold tier for unselected pages.
- [x] CTX.7 `TASK-CTX-007` (ADR-017): dynamic KV capacity. Slots grow in 1,024-position units and at least
  double, up to the instance context. Metal passes each slot's capacity in the slot table and grows with a
  blit copy; the CPU cache resizes. The tests failed first (the API was missing, then a 136-byte descriptor
  broke the 128-byte size check), then passed. A slot that grows many times is bitwise equal to a full slot
  on Metal and CPU. With `--context-size 131072` and a 4k prompt, Synapse peaks at 234 MiB (FP16 KV) and
  290 MiB (FP32 KV), where llama.cpp with the same `-c` peaks at 1,645 and 3,159 MiB.
  A request of known size reserves its slot once. Without that, a 32k prompt peaked at 686 MiB while growing;
  with it, 534 MiB, the same as before dynamic KV. The reservation test was red first (4,096 positions), then
  green.
- [x] CTX.8 `TASK-CTX-008` (ADR-018): prompt prefix reuse on the direct session (`ReusePromptPrefix`). The tests
  were red first (the API was missing), then green: bitwise equal to a cold run on CPU, the same greedy tokens on
  Metal, and invalidated by other direct work. The `prefix-reuse` experiment measures a second question about a
  long document.
- [x] BENCH.1 context sweep (`experiments ... sweep`, `sweep-report`). Every engine and KV cache type at
  4k/8k/16k/32k on three axes: tokens, memory, and speed. It feeds the README (`long-context.svg`),
  `benchmarks/README.md` run T, and the site's interactive block.
  - The harness tests (`SweepReportTests`) were written after the code, not red-first.
  - Found and fixed while measuring:
    - SwiftLM prompt-cache hits: a fresh server per sample now;
    - its early "ready" line: the model list is polled until it answers;
    - llama.cpp perf lines with timestamps;
    - the footprint-versus-RSS asymmetry, now reported as both.
  - Multi-machine comparison follows ADR-009.
- [x] REVIEW.1 independent review of the session's changes (2026-09-29, read-only agent), and fixes:
  - Prefix reuse (high): a request that failed part way left stale remembered tokens, so a retry reused
    overwritten K/V. `DirectSessionPrefix.Claim` now cuts the memory to the shared prefix before any write.
    `FailedRequestKeepsOnlyThePrefixItDidNotOverwrite` was red first (699 reused, 300 expected), then green.
  - KV pages (medium): a prompt token alone in a step (a 641-token prompt, a one-token reused tail) got
    sparse attention. `IBatchDecoder.Forward(tokens, promptStart)` now names the prompt boundary, and prompt
    tokens stay dense. `PromptTokensStayDenseEvenAloneInAStep` was red first, then green. The recorded CPU KV
    page evidence is unaffected: its scoring starts at 2,048, 7,168, and 15,360, multiples of the 64-token
    step, so no prompt token was ever alone in a step.
  - Batch scheduler (medium, older): logits were sampled after the execution gate was released, so a direct
    call could overwrite them first. Sampling now happens under the gate. No deterministic reproduction was
    found; the fix is by construction.
  - Low: CPU KV resize publishes capacity last; `AllocatedKvBytes` reads under the gate; Metal submissions end
    an open encoder on every error path, and blit regions are validated before the encoder exists; the profile
    mode keeps its command buffers in the step's GPU time; a test pins `ATT_RUN_HALVES × 8 = METAL_RUN_TOKENS`;
    a SwiftLM server that never gets ready is killed; each readiness probe has its own 2 s timeout; sweep
    summaries count failed runs (`FailedRunsStayVisibleInTheSummary`, written with the fix, not red first).
  - Verification: .NET 261 of 261, Rust 26 of 26, clippy clean.
- [ ] LONG.1 roadmap from the 2026-09-29 research (`synapse.brainstorm.md`), in order of gain per effort:
  1. A persistent prefix cache: ZoneTree metadata keyed by model, tokenizer, RoPE, KV type, and prefix hash;
     mmap payloads.
  2. Paged KV blocks: pages as the unit of growth, Quest bounds, and quantization.
  3. Quest-style decode on Metal, matched to `KvPageSelector`: absolute budgets and dense first layers.
  4. Block-sparse prefill above 32k (XAttention or FlexPrefill style), checked against dense on the quality
     suite.
  5. Head dimension 128 on the GPU, for Qwen2.5-7B-Instruct-1M (needs the owner's approval to download).
  6. GEMM counters (shader profiler) before any further kernel guesses.
- [ ] GPU.11 ideas from the llama.cpp, MLX, and vLLM study that are not yet measured here: share one
  grouped-query K/V read across the query heads of a group in decode; concurrent dispatch with two command
  buffers per token; GPU arg-max; fused QKV, gate+up+SwiGLU, and residual+norm kernels; half K/V prefill
  tiles with block-mask skipping; per-device tuning tables. Each needs a profile before it lands.
- [ ] GPU.6 wider coverage: head dimension 128 (Qwen2.5 1.5B+), QK-norm (Qwen3), and Llama/Mistral
  family adapters, which need no kernel change.
- [ ] GPU.7 batch-invariant attention: position-determined split boundaries and a sequential merge, so
  concurrent requests stay bitwise-equal to independent runs on GPU as well.
- [ ] GPU.8 `TASK-GPU-004` CUDA:
  - driver API plus NVRTC loaded dynamically, with the same ABI and kernels;
  - compile-verified remotely (a CI job with the CUDA toolkit), never by a local NVIDIA download;
  - parity suite on a real NVIDIA device. Missing hardware stays `not_run_missing_hardware`.
- [ ] GPU.9 paired benchmark cohort (ADR-005) against llama.cpp Metal at matched contexts.
