# GPU kernels

Decision: ADR-012. Plan: `gpu-kernels.plan.md`. Long-context behavior: `LongContext.md` and ADR-013.

## Requirements

- `REQ-GPU-001`: A GPU backend is selected explicitly (`--backend metal|cuda`). Its device is probed
  through a versioned C ABI, and a missing library, device, or feature fails at load with a message.
- `REQ-GPU-002`: The Metal backend runs any `DenseDecoderLayout` produced by a family adapter. It
  reproduces the pinned greedy continuation, tracks the FP32 reference within 0.2% of the logits range,
  and supports continuous batching over KV slots (ADR-007).
- `REQ-GPU-003`: Prompt runs use a tiled GEMM; decode uses split-key attention for long contexts. Both
  stay within the declared tolerance across chunking and split paths.
- `REQ-GPU-004`: The CUDA backend implements the same ABI and kernels. It is claimed only after a CUDA
  device runs the parity suite.
- `REQ-GPU-005`: An explicit FP16 KV profile (`--kv-precision f16`, runtime profile suffix `-kvf16`)
  stores keys and values in half precision and accumulates in FP32. It reproduces the pinned
  continuation and tracks the reference within 1% of the logits range. CPU backends reject it.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-GPU-001-1` the Metal probe reports an Apple GPU with unified memory; missing hardware skips as `not_run_missing_hardware` | `TEST-GPU-001-1` `MetalDeviceProbeReportsAppleGpu` |
| `AC-GPU-001-2` a missing library and a CUDA request on a machine without CUDA fail explicitly | `TEST-GPU-001-2` `MissingGpuLibraryFailsExplicitly`, `CudaBackendFailsExplicitlyWithoutDriver` |
| `AC-GPU-001-3` Rust validation rejects out-of-range, misaligned, and unimplemented layouts, accepts bias-free architectures, and plans attention work units | `cargo test -p synapse-gpu` (10 tests) |
| `AC-GPU-002-1` Metal reproduces the pinned eight-token continuation and reports its profile | `TEST-GPU-002-1` `MetalBackendMatchesPinnedContinuation`, `CliMetalBackendReportsGpuProfile` |
| `AC-GPU-002-2` Metal logits track the reference within 0.2% of the range | `TEST-GPU-002-2` `MetalLogitsTrackReferenceLogits`, `MetalPromptRunsTrackReference` |
| `AC-GPU-002-3` incremental decode tracks full prefill; concurrent requests equal independent runs | `TEST-GPU-002-3` `MetalIncrementalDecodeTracksFullPrefill`, `MetalConcurrentRequestsMatchIndependent` |
| `AC-GPU-003-1` a prompt longer than one GPU chunk tracks the managed CPU backend, and split decode attention tracks prompt-run attention | `TEST-GPU-003-1` `MetalLongPromptAcrossChunksTracksManagedCpu`, `MetalSplitDecodeTracksPromptRunAttention` |
| `AC-GPU-004-1` without the NVIDIA driver the CUDA probe and load fail as unavailable; with a driver the probe names the device (parity on NVIDIA hardware is still `not_run_missing_hardware`) | `TEST-GPU-004-1` `probe_reports_a_device_or_unavailable` (Rust), `CudaBackendFailsExplicitlyWithoutDriver` |
| `AC-GPU-005-1` FP16 KV reproduces the pinned continuation, tracks the reference within 1%, and keeps split decode within 0.2% of prompt-run attention; the CLI reports `-kvf16`; CPU backends reject FP16 KV | `TEST-GPU-005-1` `MetalFp16KvMatchesPinnedContinuationAndTracksReference`, `MetalFp16KvSplitDecodeTracksPromptRun`, `CliMetalFp16KvReportsProfile`, `Fp16KvOnCpuFailsExplicitly` |

## Implemented behavior

- `native/synapse-gpu` (Rust) contains:
  - `decoder.rs`: the brand-neutral description and its validation;
  - `ffi.rs`: the C ABI;
  - `metal/`: the device, pipelines, buffers, step encoding, and attention scheduling;
  - `metal/shaders/*.metal`: the kernels, compiled at load.
- C# side:
  - `GpuDecoderExecutor` implements `IDecoderExecutor` and `IBatchDecoder` over `DenseDecoderLayout`.
  - `NativeGpuLibrary` loads `libsynapse_gpu` from the application directory and checks the ABI.
  - `GpuDevices.Probe` reports the device.
- The Qwen2 family adapter contributes only `Qwen2DecoderLayout`, a tensor-name mapping.
- Kernels:

  | Kernel | Use |
  |---|---|
  | `synapse_q8_matvec_{1,2,4,8}` | Decode projections and logits; fixed `fma` order, independent of batch size |
  | `synapse_q8_gemm` | Prompt runs of more than 8 tokens; FP32 8×8 simdgroup matrices, 64×32 tiles |
  | `synapse_attention` | Prompt runs; K/V tile staged once for all query heads of a KV head |
  | `synapse_attention_decode` | Decode; reads K/V directly; keys split across threadgroups for long contexts |
  | `synapse_attention_reduce` | Merges decode splits |
  | `synapse_rope_kv`, `synapse_rms_norm`, `synapse_swiglu`, `synapse_embed_q8_0` | Element-wise steps |

## Current evidence

Local diagnostics only (owner M2 Pro 19-core GPU, 32 GB, macOS 27). Each figure comes from one CLI run
and carries no paired or statistical claim.

- **Decode at a short context:**
  - The first Metal version ran at about 39 tok/s. GPU trace showed 21 of 22 ms per token in the
    matrix-vector kernel.
  - The templated kernel with fixed `fma` order reached 135 tok/s for 128 tokens.
- **Prefill:**
  - At 512 tokens it measured 517 tok/s on the matrix-vector path, then 1,510 tok/s with the GEMM.
  - At 4,096 tokens it measured 527 tok/s, then 2,402 tok/s.
- **32k context:**
  - Prefill ran at 779 tok/s over the whole prompt.
  - Decode ran at 15.5 tok/s with the shared attention kernel and 62 tok/s with the dedicated decode
    kernel.
  - The cause of the slow version: 26.5 KiB of threadgroup memory allowed one threadgroup per core.
- **Later optimizations:**
  - Byte-aligned `packed_char4` code loads in the matrix-vector kernel (same `fma` order) raised
    short-context decode to 164 tok/s.
  - 32k context: 817 tok/s prefill and 75 tok/s decode with FP32 KV; 832 tok/s and 93 tok/s with
    FP16 KV.
- **131,000-token prompt (YaRN ×4, FP32 KV):**
  - Prefill ran at 151 tok/s average (869 s); decode at 33 tok/s.
  - Peak physical footprint was 3.42 GB (3 GiB KV slot); resident set 670 MB, because the weights stay
    mapped.
  - The first two attempts died with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`: one
    command buffer per step failed at 84k, one per layer at 110k. Grouped prompt-run attention completed.
  - Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-long-context-speed-diagnostic.json`.
- **llama.cpp Metal on the same machine** (`llama-bench` b29c606e2, `-fa 1`, two repetitions):
  - pp512 5,289 tok/s and tg64 157 tok/s;
  - at depth 30,000: pp 650 tok/s and tg 101 tok/s.
