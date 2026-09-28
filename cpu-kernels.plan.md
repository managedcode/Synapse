# CPU kernel performance plan

Scope: close the measured Qwen2.5 0.5B Q8_0 CPU gap to direct llama.cpp without
losing the reference oracle, greedy-token parity, or the memory-mapped RSS
advantage. Decisions are in ADR-006; requirements and tests are in
`docs/Features/CpuKernels.md`.

## Baseline (GitHub Actions run 36405097376, two threads, one sample per subject)

| | macOS ARM64 | Ubuntu x64 | Windows x64 |
|---|---|---|---|
| Synapse decode ms/token | ~151 | ~214 | ~243 |
| llama.cpp decode ms/token | ~11 | ~18 | ~22 |
| Synapse 5-token prompt TTFT ms | 804 | 1067 | 1292 |
| llama.cpp prompt eval ms | 23 | 42 | 179 |

## Steps

- [x] CPU.1 `TASK-CPU-001`: GGML-compatible activation quantizer; managed
  kernels (ARM64 sdot, x64 AVX-VNNI/AVX2, portable `Vector128`); a persistent
  worker pool with dynamic chunks; fused Q/K/V, gate/up/SwiGLU, and
  output/down with residual; the `reference|managed|native` option.
  Evidence: `ActivationQuantizerTests`, `Q8KernelTests`, `CpuWorkerPoolTests`,
  and `OptimizedQwen2Tests` pass locally on ARM64.
- [x] CPU.2 `TASK-CPU-002`: batched prefill, bitwise equal to sequential
  prefill (`BatchedPrefillMatchesSequentialPrefill`); vectorized attention;
  a RoPE table per position.
- [x] CPU.3 `TASK-CPU-003`: the Rust `synapse-kernels` cdylib with a C ABI,
  `catch_unwind`, and status codes. MSBuild builds it with cargo. It matches
  the oracle and the managed kernel. On the full weight pass it measured
  6.74 ms against 7.04 ms for the managed kernel at two threads (scratch
  microbenchmark, minimum of 40).
- [ ] CPU.4 `TASK-CPU-004`:
  - Done: tier-skipping hot methods, a page prefetch that excludes the
    embedding, allocation-free GGUF metadata skipping, `--synapse-backend` in
    the matrix, and the Rust toolchain in the performance workflow.
  - Open: remote Actions evidence on all three OSes, and x64 execution
    evidence.
  - Rejected: NativeAOT. Orleans hosting is not a NativeAOT target, and the
    local attempt failed at the Homebrew linker.
- [x] CPU.5 `TASK-CPU-005`: KV parity and continuous batching (ADR-007).
  - Local ARM64 and amd64-container runs are green:
    - `KvCacheParityTests` (9);
    - `ConcurrentGenerationTests` (6);
    - CLI `--concurrent-requests` (`CliBackendOptionTests`, 7).
  - Five concurrent 64-token requests at eight threads gave about 430
    aggregate tok/s against about 150 for one request (local diagnostic).
  - Hosted CI evidence is still open.
- [ ] CPU.6 Batched GEMM throughput: a register-blocked micro-kernel of four
  rows by four tokens, and `i8mm` where the CPU reports it. At two threads the
  batch of five is compute-bound: 162 aggregate tok/s against 118 for one
  request.
- [ ] CPU.7 Orleans cluster slice per ADR-009: node-local replicas, a request
  grain per request, and residency-aware placement.

## Local results (owner M2 Pro, two threads, median of five, diagnostic)

| Subject | Process wall | Peak RSS | Decode |
|---|---|---|---|
| Synapse `native` | 216 ms | ~555 MiB | 136.6 tok/s (end-to-end) |
| llama.cpp | 690 ms | ~1258 MiB | 133.7 tok/s (native eval) |
| LLamaSharp | 945 ms | ~1275 MiB | ~97–114 tok/s |
| dotLLM | 2227 ms | ~1190 MiB | ~9.6 tok/s |

The decode columns have different provenance (ADR-005). Five samples carry no
winner verdict.
