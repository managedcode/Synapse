namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Defined numerical failures from the scalar reference backend.</summary>
public enum ReferenceNumericalFailure
{
    /// <summary>No unmasked key exists at the requested query position.</summary>
    AllKeysMasked,
    /// <summary>An active operand contains NaN or infinity.</summary>
    NonFiniteInput,
    /// <summary>A finite computation cannot be represented as FP32.</summary>
    Fp32Overflow,
}

/// <summary>A typed failure that leaves the caller's output buffer unchanged.</summary>
/// <remarks>Creates a numerical failure with its stable code.</remarks>
public sealed class ReferenceNumericalException(ReferenceNumericalFailure failure, string message) : ArithmeticException(message)
{

    /// <summary>Stable numerical failure category.</summary>
    public ReferenceNumericalFailure Failure { get; } = failure;
}
