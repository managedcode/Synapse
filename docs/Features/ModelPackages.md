# Model packages

## Outcome

Synapse separates reproducible acquisition from executable support. The
catalog can safely download an architecture before an importer or kernel
exists, but every status surface must say which layer is available:

1. **cataloged** — immutable source files, sizes, hashes, and license exist;
2. **download verified** — the local bytes match the catalog;
3. **container indexed** — bounded GGUF/SafeTensors metadata was parsed;
4. **model imported** — config, tokenizer/template, tensors, and graph match a
   supported family;
5. **executable** — required operators and state semantics are implemented;
6. **benchmark eligible** — output parity and provenance gates pass.

Only Qwen2 Q8_0 currently reaches executable/benchmark-eligible smoke status.
SmolLM2 and the BERT embedding fixtures currently reach bounded SafeTensors
index verification. Other catalog entries are acquisition targets, not hidden
claims of runtime support.

## Catalog sets

- `smoke`: the smallest current end-to-end generation gate.
- `family-small`: small dense decoder families used while expanding import and
  tokenizer parity.
- `architecture-small`: inexpensive but structurally distinct architectures,
  currently Qwen3 and attention-free Mamba.
- `embedding-small`: two BERT-family retrieval encoders.
- `medium`: models that fit the 32 GB development Mac and expose Phi3,
  distilled Qwen2, and Mistral3 behavior under realistic memory pressure.

Native DeepSeek MLA/MoE and MiniMax hybrid/MoE are tracked separately from
their brand names. DeepSeek-R1-Distill-Qwen-1.5B is a Qwen2 checkpoint. The
current native MiniMax families are too large to be honest local smoke
fixtures. A small model is added only when it brings a distinct architecture,
not to inflate the model count.

## Trust and failure behavior

The rules are defined by `ADR-004`. Important failures happen before model
allocation: unsafe package/file path, unsupported schema, duplicate IDs/files,
non-HTTPS URL, untrusted redirect, redirect overflow, unexpected length,
digest mismatch, oversized SafeTensors header, shape overflow, overlapping
ranges, and unknown dtype.

Temporary files use a unique suffix and become visible at the destination only
after verification. An interrupted or rejected download cannot be mistaken for
a valid cached artifact.

## Source tensor decoders

`SourceFormats` decodes every stored source encoding that Synapse accepts into
FP32 reference values. On-the-fly quantization and the reference executor
consume those values.

- Floating point: F32, F64 (must fit FP32), F16, BF16, FP8 E4M3FN, and FP8
  E5M2, used for both GGUF and SafeTensors dtypes.
- GGUF blocks of 32: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, and MXFP4.
- GGUF 256-value super-blocks: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS,
  TQ1_0, and TQ2_0.

Every block is validated before anything is written. A NaN or infinite scale,
an MXFP4 or Q8_K scale that would overflow FP32, or a NaN/infinite float value
raises `SourceEncodingException` and leaves the destination unchanged.

GGUF types without a verified decoder throw `UnsupportedType` and name the
ggml type:

- Q8_1, which ggml uses for activations.
- The grid-coded IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S. They
  need ggml's lattice tables, which will be imported from pinned MIT llama.cpp
  source with attribution rather than retyped.
- NVFP4 and Q1_0.
- Integer tensor types.

FP8 decoding covers the element values only. A model's block-scale recipe
(for example a separate `weight_scale_inv` tensor) belongs to its family
adapter.

Verification:

- Golden blocks computed by hand for Q4_0, Q5_0 (high bits), Q8_0, F16/BF16,
  and FP8 anchors.
- FP8 E5M2 is compared against the upper byte of an IEEE half for every finite
  code.
- Differential tests (`GgmlReferenceDecoderTests`) call ggml's own
  `dequantize_row_*`, `ggml_fp16_to_fp32_row`, and `ggml_bf16_to_fp32_row` from
  the native library shipped with the LLamaSharp benchmark subject. They use 16
  random blocks per type. ggml serves only as the format's test oracle, never at
  runtime.
  - Multiplicative formats match bit for bit.
  - Affine formats (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) match within 2⁻²⁰ of the
    largest value, because ggml builds may fuse `a*b − c`.
  - Block sizes and type names are also checked against `ggml_blck_size`,
    `ggml_type_size`, and `ggml_type_name`.
- A deliberate sign mutation in Q4_K was caught by the differential test
  before being reverted.

## Next implementation slice

`TASK-PKG-002..005` adds the actual Synapse manifest/chunk format, tensor source
descriptors, repo-owned BPE/chat templates, SafeTensors conversion, ZoneTree
content identity cache, and atomic compiler/export. It must not execute remote
model code or introduce Python/Node tooling.
