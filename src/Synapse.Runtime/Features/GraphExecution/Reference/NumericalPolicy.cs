namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Named absolute and relative tolerances for bounded numerical comparisons.</summary>
/// <param name="AbsoluteTolerance">Maximum permitted absolute difference near zero.</param>
/// <param name="RelativeTolerance">Maximum permitted difference relative to the oracle magnitude.</param>
public readonly record struct NumericalPolicy(double AbsoluteTolerance, double RelativeTolerance)
{
    /// <summary>Initial FP32 tolerance for bounded tiny tensors.</summary>
    public static NumericalPolicy Fp32 { get; } = new(1e-5, 1e-4);

    /// <summary>Initial FP16 tolerance for bounded tiny tensors.</summary>
    public static NumericalPolicy Fp16 { get; } = new(5e-3, 5e-3);

    /// <summary>Initial BF16 tolerance for bounded tiny tensors.</summary>
    public static NumericalPolicy Bf16 { get; } = new(2e-2, 2e-2);

    /// <summary>Checks one computed value against a finite oracle result.</summary>
    public bool IsWithinTolerance(float actual, double expected) =>
        float.IsFinite(actual) && double.IsFinite(expected) &&
        Math.Abs((double)actual - expected) <=
        AbsoluteTolerance + (RelativeTolerance * Math.Abs(expected));
}
