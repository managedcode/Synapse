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
