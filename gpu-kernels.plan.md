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
- [ ] GPU.10 prompt-run attention occupancy. The kernel uses 26.5 KiB of threadgroup memory, so one
  threadgroup fits per core. Candidates: half K/V tiles for the FP16 profile, 16-key tiles, and 16 query
  rows per simdgroup. At 32k it roughly matches llama.cpp (~1.6 TFLOPS effective), but on the identical
  120k pass-key prompt llama.cpp prefilled in 288 s against our 585 s. This is the largest measured gap.
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
- [ ] QUAL.3 correct the 120k llama.cpp pass-key control: llama.cpp `-f` drops the prompt's trailing
  newline, so that control evaluated 119,990 tokens, not the 119,991 recorded as identical. Rerun it with
  the compensated prompt file and fix the evidence.
- [ ] QUAL.4 `TASK-CTX-006` (ADR-016): query-aware KV page activation, the owner's "smart context" idea as
  partial activation over the context. Done: the C# reference selection on the managed and native CPU
  backends, and `--kv-pages` in `generate` and `score`. The tests were red (types missing), then 5 of 5
  green, including bitwise equality with dense for a covering budget. Open: the CPU decode-mode quality
  runs (key bound against the random control), then the Metal kernels matched to the C# selection, then a
  cold tier for unselected pages.
- [ ] BENCH.1 context sweep (`experiments ... sweep`): every engine and KV cache type at 4k, 8k, 16k, and
  32k on three axes (tokens, memory, speed), feeding the README, `benchmarks/README.md`, and the site.
  Multi-machine comparison follows ADR-009.
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
