# ADR-021: GGUF K-quant weights (Q4_K and Q6_K)

Status: Accepted. Date: 2026-09-29. Extends ADR-004 (model packages) and ADR-012 (GPU backends).

## Context

The owner fetched Qwen2.5-7B-Instruct-1M in both Q8_0 and Q4_K_M. Q4_K_M mixes two encodings:
- most matrices are Q4_K: 256-value super-blocks in 144 bytes, 4.5 bits a weight;
- the sensitive ones are Q6_K (the output projection, and V and the feed-forward down projection in some
  layers): 210 bytes, 6.56 bits a weight.

In llama.cpp it peaks at 4,742 MiB against 8,096 for Q8_0 and decodes at 29.2 against 18.6 tokens/s. It is the
largest memory and speed lever measured so far, larger than dropping layers (ADR-019).

Synapse executes Q8_0 only; `SourceEncodings` already decodes every K-quant exactly, as ggml's
`dequantize_row_*` does.

## Decision

- **Contract.**
  - `WeightEncoding` gains `GgmlQ4K` and `GgmlQ6K`.
  - `StorageDataType.BlockQ4` stores Q4_K, and a new `BlockQ6` stores Q6_K.
  - The GGUF reader sizes Q4_K and Q6_K tensors, which need a multiple of 256 values.
  - The graph verifier accepts the pairs `GgmlQ4K`/`BlockQ4` and `GgmlQ6K`/`BlockQ6`.
- **Reference backend.**
  - A non-Q8_0 matrix row (Q4_K, Q6_K, or F32) is decoded exactly into FP32 with the `SourceEncodings`
    decoder, then dotted with the input in index order.
  - An F32 model holding the dequantized values therefore produces bitwise the same logits, which is the test
    oracle.
  - The Q8_0 path is unchanged.
- **Managed and native CPU kernels** reject K-quant matrices at load with `NotSupportedException`, until they
  have K-quant dot products.
- **Metal.**
  - Every matrix carries its own encoding.
  - `DecoderLayerOffsets` gains `encodings`: 4 bits per matrix, in the order Q, K, V, O, gate, up, down.
  - The descriptor's `matrix_encoding` becomes the token-embedding encoding, and its former `pad` becomes the
    output encoding.
  - The C ABI version becomes 3.
  - One projection dispatch holds one encoding. Q, K, and V split into two dispatches when V's encoding
    differs.
  - The matrix-vector, GEMM, and embedding kernels dequantize exactly as the reference does and accumulate in
    FP32.
- **CUDA** keeps Q8_0 and rejects K-quant descriptors explicitly.

## Consequences

- The Q4_K_M 7B maps 4.7 GB instead of 8.1 GB, and decode reads 42% fewer bytes a token.
- K-quant numerics follow ggml's decode. Synapse compares with llama.cpp on the same file, and quality versus
  Q8_0 is measured, never assumed.
- Tests:
  - `KQuantReferenceEqualsDequantizedFp32` (bitwise);
  - `KQuantWeightsFailExplicitlyOnCpuKernels`;
  - `MetalMatchesReferenceOnKQuantWeights`;
  - Rust range and segment checks for K-quant sizes.

## Measured (2026-09-29)

- **Memory.** Qwen2.5-7B-Instruct-1M Q4_K_M peaks at 4,233–4,248 MiB in Synapse, against 4,846–4,851 in
  llama.cpp and 7,210–7,252 for Synapse Q8_0.
- **Text.** The pinned prompt's 16 tokens are the same text as llama.cpp's.
- **Quality.** Perplexity on the pinned tokens is 11.93, against 12.16 for Q8_0.
- **Decode: 0.64× llama.cpp's** (17.4–18.2 against 27.0–29.3 tokens/s).
  - The K-quant matrix-vector kernel is latency-bound.
  - One row per simdgroup with only a `simd_sum` reduction was the fastest layout tried, going from 12.4 to
    19.0 tokens/s on a chat prompt.
  - Decoding each chunk once for every token helped multi-token steps.
- **Prefill** of 3,528 tokens takes 20 s against 12 s: the K-quant GEMM stages without load pipelining.
- **Bug fixed along the way.** `SourceEncodings` built its GGUF decoder table before the float decoders it
  references had been initialized, so F32, F16, BF16, and F64 resolved to null. It is fixed and pinned by
  `GgufFloatTypesResolveToTheirDecoders`.
