namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>Stable categories of quantized-weight failures.</summary>
public enum QuantizationFailure
{
    /// <summary>A source weight is NaN or infinity.</summary>
    NonFiniteWeight,
    /// <summary>A group scale cannot be represented by its storage type.</summary>
    ScaleOverflow,
    /// <summary>A stored scale is NaN, infinite, negative, or negative zero.</summary>
    InvalidScale,
    /// <summary>A stored code uses the reserved value.</summary>
    ReservedCode,
    /// <summary>A padding element or a zero-scale group stores a non-zero code.</summary>
    NonCanonicalPadding,
}

/// <summary>A typed quantization failure that leaves the caller's destination unchanged.</summary>
/// <remarks>Creates a quantization failure with its stable code.</remarks>
public sealed class QuantizationException(QuantizationFailure failure, string message) : FormatException(message)
{

    /// <summary>Stable failure category.</summary>
    public QuantizationFailure Failure { get; } = failure;
}
