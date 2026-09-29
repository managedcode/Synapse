# Quantization

## Implemented behavior

The C# reference path provides grouped row-major weight encodings of any width
from 2 to 8 bits, plus ternary, and the tools that decide where to use them.
None of it is wired into model execution yet; see `elastic-inference.plan.md`
E2–E5.

- `syn.q{bits}.symmetric.g{group}.v1` (`SymmetricGroupCodec`) works for any
  `bits` from 2 to 8 and any `group` that is a multiple of 8 up to 1024. Each
  group stores an FP16 scale followed by `group * bits / 8` bytes of codes,
  packed least-significant-bit first.
  - Codes are `q + 2^(bits-1)` with `|q| ≤ 2^(bits-1) − 1`; code 0 is
    reserved.
  - The scale is `max|w| / (2^(bits-1) − 1)` rounded up to the next FP16
    value, so no code is clamped and the error stays within half a scale step.
  - Codes round ties to even. Padding and zero groups store `q = 0`.
  - `WeightCodecs.TryGet` resolves any valid ID back to its codec.
- `syn.q4.symmetric.g64.v1` (`SynQ4BlockCodec`, spec §6.1) is the 4-bit,
  64-element member: 34 bytes per group (4.25 bits per weight), even elements
  in the low nibble. It is not GGUF `Q4_K`.
- `syn.ternary.absmean.g64.v1` (`TernaryBlockCodec`) stores each group as an
  FP16 absmean scale followed by 16 bytes of 2-bit codes (0, +1, −1, and a
  reserved value), which is 2.25 bits per weight.
  - This is the BitNet b1.58 rule applied per group after training, so it is
    lossy by design.
  - Its linear uses additions only inside each group.
- Both decoders validate every group before writing: the scale must be finite
  and non-negative, reserved codes and non-zero padding are rejected, and so
  are non-zero codes in a zero-scale group. Encoders reject non-finite weights
  and scale overflow before writing.
- `WeightSensitivity` measures activation-aware relative output distortion
  (spec §6.2). This is a ranking heuristic, not a quality proof.
- `PrecisionBudgetSelector` assigns one encoding per tensor under a byte
  budget.
  - It is greedy, demoting the least important tensor first.
  - It honors the device's supported encodings and pinned tensors, and it
    requires an explicit approximation opt-in.
  - It is monotone in the budget, so a larger budget only promotes tensors.
  - `Diff` lists the promotions and demotions between two profiles.
  - `ProfileHash` identifies the effective weights.

## Acceptance mapping

| Scenario | Test |
|---|---|
| AC-QNT-001-1 golden bytes | `Q4GoldenBytesMatch`, `OddWidthGoldenBytesMatch`, `TernaryGoldenBytesMatch` |
| Any width 2–8 bits, any group size | `EveryBitWidthRoundTripsWithinHalfScale`, `EveryBitWidthLinearMatchesDequantizedReference`, `MoreBitsNeverIncreaseDistortion`, `InvalidParametersRejected`, `WeightCodecsResolveById` |
| Per-tensor options for any set of codecs | `MeasuredOptionsCoverAnyBitWidthInSizeOrder`, `EqualSizeOptionsKeepLowerDistortion`, `EncodingsNotSmallerThanSourceAreDropped` |
| AC-QNT-001-2 invalid scale | `InvalidScaleRejected`, `TernaryRejectsReservedCodeInvalidScaleAndPadding` |
| AC-QNT-001-3 size accounting | `SizeAccountingExact`, `TernarySizeAccountingExact` |
| Q4 linear without a dequantized tensor (TASK-QNT-002 reference) | `Q4LinearMatchesDequantizedReference` |
| TASK-QNT-003 greedy budgeted profile | `LeastImportantTensorIsDemotedFirst`, `InfeasibleBudgetRejectedWithRequirement`, `ApproximationRequiresExplicitPolicy` |
| Automatic restore by budget | `LargerBudgetOnlyPromotes`, `PromotionRestoresMostImportantDemotedTensorsFirst` |

Not yet covered: the calibration/test split enforcement (AC-QNT-003-1) and
ablation validation (AC-QNT-003-3) require real-model activations from region
execution.

## GGUF K-quant weights in execution (ADR-021)

- **Requirement `REQ-QNT-005`:** Q4_K and Q6_K GGUF matrices, as in Q4_K_M files, execute on the reference and
  Metal backends.
  - The reference path decodes each row exactly as ggml does, so an F32 model holding the decoded values gives
    bitwise the same logits.
  - Metal dequantizes in its matrix-vector, GEMM, and embedding kernels. It tracks the reference within 0.2% of
    the logits range with FP32 KV, and within 1% with FP16 KV.
  - The managed and native CPU kernels reject K-quant matrices with an ADR-021 message, and CUDA rejects them as
    unavailable.
- **Profile names.** The runtime profile names the encodings, for example `metal-qwen2-q4_k+q6_kxf32`.

| Criterion | Test |
|---|---|
| `AC-QNT-005-1` K-quant reference logits equal the dequantized F32 model bitwise | `TEST-QNT-005-1` `KQuantReferenceEqualsDequantizedFp32` |
| `AC-QNT-005-2` managed and native CPU kernels reject K-quants explicitly | `TEST-QNT-005-2` `KQuantWeightsFailExplicitlyOnCpuKernels` |
| `AC-QNT-005-3` Metal tracks the reference on a Q4_K_M-shaped model, and the profile names the encodings | `TEST-QNT-005-3` `MetalMatchesReferenceOnKQuantWeights` |
| `AC-QNT-005-4` K-quant sizes and whole super-blocks are validated, and host row counts match the shader instances | `TEST-QNT-005-4` `plan_sizes_kquant_matrices_by_their_blocks`, `kquant_rows_must_hold_whole_super_blocks`, `kquant_matvec_rows_match_the_shader_instances` (Rust) |
| `AC-QNT-005-5` every GGUF float type resolves to its decoder | `TEST-QNT-005-5` `GgufFloatTypesResolveToTheirDecoders` |

**Evidence.** Run Y in `benchmarks/README.md`
(`2026-09-29-m2-pro-qwen2.5-7b-1m-k-quant-diagnostic.json`). On Qwen2.5-7B-Instruct-1M Q4_K_M:
- Synapse writes llama.cpp's text for the pinned prompt.
- It peaks at 4,233–4,248 MiB, against 4,846–4,851 for llama.cpp and 7,210–7,252 for Synapse Q8_0.
- Decode is 17.4–18.2 tokens/s against llama.cpp's 27.0–29.3.
- Perplexity is 11.93 against 12.16 for Q8_0 on the same tokens.
