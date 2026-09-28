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
public sealed class ReferenceNumericalException : ArithmeticException
{
    /// <summary>Creates a numerical failure with its stable code.</summary>
    public ReferenceNumericalException(ReferenceNumericalFailure failure, string message)
        : base(message)
    {
        Failure = failure;
    }

    /// <summary>Stable numerical failure category.</summary>
    public ReferenceNumericalFailure Failure { get; }
}
