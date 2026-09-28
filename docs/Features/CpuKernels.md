# CPU kernels

## Requirements

- `REQ-CPU-001`: The Q8_0 dense decoder has an optimized CPU path whose
  activation quantization, integer block dots, and accumulation follow ADR-006.
  It must keep greedy-token parity on the pinned model.
- `REQ-CPU-002`: Prompt evaluation shares one weight pass per prompt chunk
  and is bitwise identical to sequential evaluation.
- `REQ-CPU-003`: An in-process Rust kernel is selectable through a versioned C
  ABI. It fails explicitly when unavailable and matches the managed kernel.
- `REQ-CPU-004`: Kernel backend, thread cap, and runtime profile are explicit
  in the CLI and in benchmark evidence.
- `REQ-CPU-005`: KV slots stay exact under incremental decode. Concurrent
  requests share batched steps (ADR-007) without changing any request's
  tokens.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-CPU-001-1` quantizer matches GGML Q8_0 semantics, including ties, zero blocks, and FP16 scales | `TEST-CPU-001-1` `ActivationQuantizerMatchesGgmlContract` |
| `AC-CPU-001-2` every available managed ISA path matches an FP64 oracle on real GGUF rows | `TEST-CPU-001-2` `ManagedKernelMatchesFp64Oracle` |
| `AC-CPU-001-3` the worker pool runs every chunk exactly once, reuses its threads, and rethrows worker failures | `TEST-CPU-001-3` `WorkerPoolRunsEveryChunkOnce`, `WorkerPoolRethrowsWorkerFailure` |
| `AC-CPU-001-4` the managed backend reproduces the pinned eight-token continuation | `TEST-CPU-001-4` `ManagedBackendMatchesPinnedContinuation` |
| `AC-CPU-002-1` batched prefill logits equal sequential prefill logits bitwise | `TEST-CPU-002-1` `BatchedPrefillMatchesSequentialPrefill` |
| `AC-CPU-003-1` the native library loads with the expected ABI, and its kernel matches the managed kernel | `TEST-CPU-003-1` `NativeKernelMatchesManagedKernel` |
| `AC-CPU-003-2` the native backend reproduces the pinned continuation, and an unavailable library fails explicitly | `TEST-CPU-003-2` `NativeBackendMatchesPinnedContinuation`, `MissingNativeLibraryFailsExplicitly` |
| `AC-CPU-003-3` Rust kernels match a scalar oracle, and the FFI rejects invalid arguments without panicking | `cargo test -p synapse-kernels` |
| `AC-CPU-004-1` CLI `--backend` selects the backend, reports it in the runtime profile, and rejects unknown names | `TEST-CPU-004-1` `CliBackendOptionSelectsRuntimeProfile` |
| `AC-CPU-005-1` incremental KV decode equals full prefill bitwise; stale slot contents never leak; multi-chunk prompts equal sequential; context and vocabulary bounds are exact | `TEST-CPU-005-1` `IncrementalDecodeMatchesFullPrefill`, `GenerationRepeatsAfterLongerPrompt`, `LongPromptAcrossChunksMatchesSequential`, `ContextCapacityIsEnforcedExactly`, `KvCacheKeepsHeadsSeparateAndRejectsOutOfRangePositions` |
| `AC-CPU-005-2` concurrent requests equal independent runs on every backend; decode is never starved by a long prefill | `TEST-CPU-005-2` `RaggedBatchMatchesIndependent`, `LongPrefillDoesNotStarveDecode` |
| `AC-CPU-005-3` a canceled request completes as canceled and its reused slot produces exact output; CLI concurrency reports per-request and aggregate results | `TEST-CPU-005-3` `CancelledSlotReusedSafely`, `CliRunsConcurrentRequestsThroughOneModel`, `CliRejectsInvalidConcurrency` |

## Implemented behavior

- `Q8ActivationQuantizer` quantizes each activation row with the GGML Q8_0
  rule. `Q8ActivationBuffer` holds pinned rows for up to one prompt chunk.
- There are three managed kernels, and `ManagedQ8Kernel.CreateBest()` picks
  the fastest one the CPU supports.

  | Kernel | Instructions | Short rows | Long rows |
  |---|---|---|---|
  | `Q8ArmKernel` | ARM64 `sdot` | Four rows per activation load | One sequential stream, four accumulators |
  | `Q8X86Kernel` | AVX2, or AVX-VNNI when present | Four rows per activation load | One stream, two accumulators |
  | `Q8PortableKernel` | `Vector128` widening | One row at a time | One row at a time |

  A row counts as long at 64 blocks or more. That threshold was measured on
  the down projection.
- `NativeQ8Kernel` loads `synapse_kernels` from the application directory and
  checks its ABI version. The Rust crate `native/synapse-kernels` has three
  kernels:
  - NEON `dotprod`, with the same row strategy as the managed ARM64 kernel;
  - AVX2/FMA;
  - a safe scalar oracle.
- `CpuWorkerPool` keeps `threads - 1` persistent workers, and the caller acts
  as worker 0. `ChunkedRowWork` splits a region into dynamic row chunks. A
  chunk has at most `16384 / tokens` rows, so its partial buffers stay bounded.
- `Qwen2CpuExecutor` runs four fused regions per layer plus logits. Prompt
  chunks of up to 64 tokens share one pass over each weight row.
- Loading has two optimizations:
  - While the caller builds and verifies the graph, the worker pool prefaults
    the densely read tensors. The sparse token embedding is excluded, so
    resident memory stays at the used weights.
  - GGUF metadata strings that the runtime does not use are skipped without
    allocating.

## Current boundary and evidence

- The optimized paths cover the Qwen2 GGUF Q8_0 decoder on CPU. Q4, Metal, KV
  paging, and speculative decoding are not included.
- All backends reproduce the pinned eight-token continuation.
- At 32 tokens, `managed` and `native` produce identical tokens to each other.
  They diverge from the FP32 reference and llama.cpp at token 22. At that
  step the FP32 reference itself separates its top two tokens by only 0.0115
  logits (4126 at 15.2717, 9806 at 15.2602). Q8_0 activations reorder them by
  0.0163. Earlier steps keep margins such as 2.91.
- This is a near-tie, not a kernel defect. Still, a long exact-continuation
  quality gate can mark these paths as ineligible. Two remedies remain open:
  locked-continuation speed runs, or an FP32-activation parity profile.
- x64 execution evidence comes from a local `linux/amd64` Docker container
  (emulated CPU with `avx2` and `fma`, SDK 10.0.302). There the CLI selected
  `managed-x64-avx2` and `native-x64-avx2`, and both reproduced the pinned
  eight tokens. These suites passed:
  - Rust `synapse-kernels`: 7/7, with the AVX2 oracle parity case;
  - `Q8KernelTests`, `OptimizedQwen2Tests`, `KvCacheParityTests`,
    `ConcurrentGenerationTests`, and `CliBackendOptionTests`.

  The AVX-VNNI path has not run on any hardware yet. Emulation timings are not
  performance evidence. Hosted CI has not run this change set.
- Local diagnostic matrices are recorded in `benchmarks/results/`:
  - `2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json`
  - `…-managed-2thread-smoke.json`
  - `…-native-8thread-smoke.json`

  Each has 3 warm-ups and 5 measurements. They are not paired release
  evidence.
