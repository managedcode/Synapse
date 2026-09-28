# ADR-006: CPU kernel backends and the native kernel boundary

Status: Accepted. Date: 2026-09-28.

## Context

The first four-subject diagnostic matrix (GitHub Actions run `36405097376`,
two compute threads) showed that Synapse spent 150–240 ms per decoded Qwen2.5
0.5B Q8_0 token while direct llama.cpp spent 11–22 ms. The difference was not
model loading: outside generation, Synapse used 155–350 ms and llama.cpp
360–900 ms. The profile points to one managed hotspot. The Q8_0 linear
converted every weight to FP32 in a scalar loop, it dispatched one
`Parallel.For` delegate per output row, it ran 169 parallel regions per token,
and it evaluated prompt tokens one at a time.

## Decision

Synapse has three explicit CPU kernel backends for GGUF Q8_0 dense models.

| Backend | Owner | Numeric contract |
|---|---|---|
| `reference` | C# | FP32 activations times dequantized Q8_0 weights, scalar, sequential prompt. This is the permanent oracle path. |
| `managed` | C# | Q8_0 weights times Q8_0-quantized activations, SIMD, fused, batched prefill. |
| `native` | Rust `synapse_kernels` via C ABI | Same contract as `managed`, with only the integer-dot matrix kernel in Rust. |

- **Selection is explicit.** `ModelLoadOptions.KernelBackend` and CLI
  `--backend` choose one backend. The default is `managed` because it runs on
  every .NET platform: it uses `Dp` (ARM64 sdot), AVX-VNNI, or AVX2 when the
  CPU has them, and portable `Vector128` otherwise. A `native` request fails
  with `NotSupportedException` when the library is missing or reports a
  different ABI version. It never falls back silently.
- **Activation quantization matches the GGML Q8_0 contract.** Each 32-value
  block has `d = max|x| / 127`, a stored FP16 `d`, and codes
  `round_half_even(x * (1/d))` computed with the unrounded `d`. A zero block
  stores zero codes and a zero scale. Integer block dots are exact. The scale
  product and the cross-block accumulation are FP32, and the summation order
  may differ by ISA. Parity is established against an FP64 oracle over the
  same quantized inputs and by greedy-token parity on the pinned model.
- **Weights stay memory-mapped.** No weight is repacked, copied, or converted
  at load, so resident memory stays close to the GGUF payload.
- **One persistent worker pool per model.** It follows the thread cap:
  `threads = 1 + workers`, and the calling thread computes. Work is split into
  dynamic row chunks claimed with one atomic counter. Idle workers spin for a
  bounded time, then block. A worker exception is rethrown on the caller.
- **Fused per-layer regions.** Q/K/V with biases, gate/up with SwiGLU, and
  output/down with the residual add run as four parallel regions per layer.
  The logits region is the fifth. The logits region runs only for the final
  prompt token.
- **Batched prefill.** A prompt chunk shares one pass over each weight row.
  Every row and token dot uses the same kernel as decode, so batched and
  sequential prefill are bitwise identical.

## Native boundary

- `native/synapse-kernels` builds as a `cdylib` for .NET and an `rlib` for
  Rust tests. The `Synapse.Runtime` build runs
  `cargo build --release --locked -p synapse-kernels` and copies the library
  next to the managed assemblies. The library is loaded only from the
  application directory, never from `PATH`.
- The exported surface is small and versioned:
  - `synapse_kernels_abi_version`
  - `synapse_kernels_capabilities`
  - `synapse_q8_0_matmul`

  Invalid arguments return non-zero status codes. Kernel bodies run inside
  `catch_unwind`, so no panic crosses the ABI.
- This crate alone replaces the workspace `unsafe_code = "forbid"` lint with
  `deny`. `unsafe` is allowed only in the FFI and ISA modules. Every unsafe
  block carries a `SAFETY:` comment, which `clippy::undocumented_unsafe_blocks`
  enforces. The C# caller validates tensor shapes and owns every buffer for
  the duration of the call.
- Threading stays in C#. Each worker calls the native kernel once per row
  chunk, so the P/Invoke transition is amortized over at least 16 rows.

## Consequences

- Building the .NET solution now requires the pinned Rust toolchain.
  `verify.yml` and `performance.yml` install it before the .NET build.
- Model runtime profiles name the backend. The benchmark matrix passes
  `--synapse-backend` explicitly, so raw evidence records which kernel ran.
- The scalar reference path remains tested.
- A performance claim still needs the ADR-005 evidence contract. Faster
  kernels do not replace paired raw samples and quality parity.
