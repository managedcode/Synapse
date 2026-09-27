namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Physical representation of a tensor in storage.</summary>
public enum StorageDataType
{
    /// <summary>IEEE 754 single precision.</summary>
    Fp32,

    /// <summary>IEEE 754 half precision.</summary>
    Fp16,

    /// <summary>Brain floating-point 16-bit representation.</summary>
    Bf16,

    /// <summary>Versioned block Q8 storage.</summary>
    BlockQ8,

    /// <summary>Versioned block Q4 storage.</summary>
    BlockQ4,

    /// <summary>Signed 32-bit integer storage.</summary>
    I32,

    /// <summary>Boolean storage.</summary>
    Bool,
}

/// <summary>Arithmetic precision used by a kernel.</summary>
public enum ComputeDataType
{
    /// <summary>IEEE 754 single precision.</summary>
    Fp32,

    /// <summary>IEEE 754 half precision.</summary>
    Fp16,

    /// <summary>Brain floating-point 16-bit arithmetic.</summary>
    Bf16,

    /// <summary>Signed 32-bit integer arithmetic.</summary>
    I32,

    /// <summary>Boolean operations.</summary>
    Bool,
}

/// <summary>Precision used to accumulate reductions.</summary>
public enum AccumulatorDataType
{
    /// <summary>IEEE 754 single precision accumulation.</summary>
    Fp32,

    /// <summary>IEEE 754 double precision accumulation.</summary>
    Fp64,

    /// <summary>Signed 32-bit integer accumulation.</summary>
    I32,
}

/// <summary>Explicit storage, arithmetic, and accumulation precision.</summary>
/// <param name="Storage">Physical storage representation.</param>
/// <param name="Compute">Arithmetic precision.</param>
/// <param name="Accumulator">Reduction precision.</param>
public readonly record struct NumericType(
    StorageDataType Storage,
    ComputeDataType Compute,
    AccumulatorDataType Accumulator);

/// <summary>A fixed dimension or a named dimension with finite bounds.</summary>
/// <param name="Symbol">Symbol name, or <see langword="null"/> for a fixed dimension.</param>
/// <param name="Minimum">Inclusive lower bound.</param>
/// <param name="Maximum">Inclusive upper bound.</param>
public readonly record struct ShapeDimension(string? Symbol, long Minimum, long Maximum)
{
    /// <summary>Creates a fixed dimension.</summary>
    public static ShapeDimension Fixed(long value) => new(null, value, value);

    /// <summary>Creates a named, finitely bounded dimension.</summary>
    public static ShapeDimension Bounded(string symbol, long minimum, long maximum) =>
        new(symbol, minimum, maximum);

    /// <summary>Whether this dimension is symbolic.</summary>
    public bool IsSymbolic => Symbol is not null;
}

/// <summary>Immutable tensor shape whose dimensions are validated by the graph verifier.</summary>
public sealed class TensorShape
{
    /// <summary>Creates a shape from fixed or bounded dimensions.</summary>
    public TensorShape(params ShapeDimension[] dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        Dimensions = Array.AsReadOnly([.. dimensions]);
    }

    /// <summary>Dimensions in row-major logical order.</summary>
    public IReadOnlyList<ShapeDimension> Dimensions { get; }

    /// <summary>Number of logical dimensions.</summary>
    public int Rank => Dimensions.Count;
}
