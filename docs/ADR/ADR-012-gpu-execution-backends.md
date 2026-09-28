# ADR-012: GPU execution backends (Metal first, CUDA next)

Status: Accepted. Date: 2026-09-28.

## Context

The owner asked for Apple GPU support, NVIDIA CUDA support, strong performance, and long contexts (30k,
40k, 128k). The CPU backends (ADR-006) decode Qwen2.5-0.5B Q8_0 at CPU speed, and their attention re-reads
every K/V row per query head, so a 32k prefill takes minutes on CPU. llama.cpp, MLX, and DwarfStar
(antirez/ds4) reach GPU speed with a small set of hand-written kernels: quantized matrix-vector for
decode, tiled simdgroup/tensor-core GEMM for prompts, flash attention with online softmax, and split-key
decode attention. All of them are studied as designs only; no source is copied.

Synapse must stay model-agnostic and format-neutral: ADR-004 makes a Synapse package compiler the long-term
source of weights, and GGUF is only one adapter.

## Decision

- **Backends are explicit.** `KernelBackend` (renamed from `CpuKernelBackend`) gains `Metal` and `Cuda`;
  CLI `--backend metal|cuda`. A missing library, device, feature, or unsupported model shape fails at
  load with `NotSupportedException`. Nothing falls back to CPU silently.
- **One native library, one brand-neutral ABI.** `native/synapse-gpu` builds `libsynapse_gpu` (cdylib)
  with a versioned C ABI: `synapse_gpu_abi_version`, `synapse_gpu_probe`, `synapse_gpu_last_error`, and
  `synapse_gpu_decoder_create|forward|destroy`. The ABI describes a pre-norm dense decoder, not a model
  family:
  - byte offsets into one caller-mapped weight file;
  - optional Q/K/V biases, marked `NO_TENSOR` when absent;
  - declared matrix encoding, RoPE layout, and activation;
  - C#-computed RoPE tables.

  Every entry point validates ranges, catches panics, and returns a status code plus a message.
- **Model agnosticism lives in C#.** The executor is `GpuDecoderExecutor`; it consumes `DenseDecoderLayout`
  built from Model IR types (`WeightSourceRange`, `WeightEncoding`). A family adapter only maps its tensor
  names onto that layout (`Qwen2DecoderLayout` today). Llama, Mistral, and SmolLM2 are the same layout
  without biases. Qwen3 needs QK-norm, and larger Qwen2.5 models need head dimension 128. Until the
  kernels implement those, they are rejected as `Unavailable`. When the ADR-004 package compiler lands,
  a second adapter maps its manifest onto the same layout, and GPU code does not change.
- **Ownership.** C# owns scheduling, sampling, session and slot ownership (ADR-007), validation, and RoPE
  frequencies (ADR-013). The native side owns device buffers, pipelines, command encoding, and the KV slot
  memory it is leased. It never decides which session owns a slot.
- **Metal host.** Rust with `objc2` (MIT) and `objc2-foundation`/`objc2-metal` (Zlib, Apache-2.0, or MIT).
  - Kernels are compiled from source at load with `MTLMathMode.safe` and Metal 3.1, so the build needs
    no Metal toolchain.
  - Requirements: an Apple7+ GPU (simdgroup matrices) and macOS 15+.
  - The mapped weight file is wrapped once with `newBufferWithBytesNoCopy`, so weights are neither copied
    nor repacked, and resident memory stays near the file size.
  - KV slots are shared buffers, allocated lazily, zero-filled, and padded by one attention tile.
    Kernels reach them through GPU addresses in an argument struct.
  - Each step is one command buffer, except long prompts. The macOS GPU watchdog
    (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) killed a 512-token chunk at 84k keys in one
    buffer, and again at 110k with one buffer per layer. A long step therefore gets a submission per
    layer, and prompt-run attention goes out in groups of at most 2^24 query-key pairs.
  - Metal and CUDA share one backend-neutral step encoder (`step.rs`, trait `StepBackend`): the same
    kernels, arguments, and geometry, where kernel parameters follow the Metal buffer indices.
- **Numeric profile `metal-<arch>-q8_0xf32`.** Q8_0 weights are dequantized exactly (`d * q`). Activations,
  accumulation, KV, and softmax are FP32. There is no Q8 activation quantization, so Metal tracks the FP32
  `reference` oracle more closely than `managed`. Two kernel paths produce the projections:
  - Decode tokens use a batch-invariant matrix-vector kernel: explicit `fma` order, templated for 1, 2,
    4, or 8 tokens.
  - Prompt runs of more than 8 tokens use a tiled FP32 simdgroup-matrix GEMM, with 64 rows × 32 tokens
    per tile and one Q8_0 block per K step.

  Prompt-run attention stages a K/V tile shared by the query heads of one KV head. Decode attention reads
  K/V directly with little threadgroup memory, and a long context splits its keys and merges them in a
  reduce kernel.
- **FP16 KV profile.** `KvCachePrecision.Fp16` (CLI `--kv-precision f16`, profile suffix `-kvf16`) stores
  K and V in half precision. Loads convert or multiply half tiles into FP32 simdgroup accumulators
  (mixed-precision `simdgroup_multiply_accumulate`). It halves KV memory and attention bandwidth: the slot
  for a 131,072-token context shrinks from 3 GiB to 1.5 GiB. It is an explicit numerical profile with a 1%
  reference tolerance, never an implicit default. CPU backends and, for now, CUDA reject it as unsupported.
- **Parity contract.** Metal must reproduce the pinned greedy continuation and stay within a declared
  logits tolerance of `reference` (0.2% of the logits range) and of itself across chunking and split
  paths. It is not bitwise-equal to CPU backends, and across the matrix-vector and GEMM paths it is not
  batch-invariant. A token that is decoded alone uses the same kernel as one decoded in a batch.
- **CUDA.** Same ABI, the same step encoder, and ports of the same kernels in FP32 CUDA C (CUDA cores,
  no tensor cores yet):
  - Host through the CUDA driver API, with `libloading` (ISC) for `libcuda`/`nvcuda.dll` and NVRTC.
  - Kernels compiled to PTX at load for the device compute capability (6.0+), so the build needs no
    `nvcc`.
  - Weights are copied to device memory once.
  - A missing driver or device is `Unavailable`.

  The owner has no NVIDIA hardware, and nothing NVIDIA is downloaded locally. The kernels have passed
  only a local C++ syntax check with a CUDA shim. Until a CUDA device, or at least a remote NVRTC
  compile, runs the parity suite, the backend is not claimed; tests without NVIDIA hardware report
  `not_run_missing_hardware`.

## Consequences

- Building the solution builds both Rust cdylibs (`synapse-kernels`, `synapse-gpu`). The new crates.io
  dependencies are macOS-only and pinned in `Cargo.lock`.
- GPU tests skip with `not_run_missing_hardware` off Apple silicon and on devices below Apple7; on Apple
  silicon a probe failure fails the test.
- Performance claims still need ADR-005 paired raw evidence. The numbers in
  `docs/Features/GpuKernels.md` are local diagnostics.
- Open work is tracked in `gpu-kernels.plan.md`: FP16 KV profile, GPU-friendly repacked weights in the
  Synapse package format, head dimension 128, QK-norm, batch-invariant attention splits, and CUDA.
