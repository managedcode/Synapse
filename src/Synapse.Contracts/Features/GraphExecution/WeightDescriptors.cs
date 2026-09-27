namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Physical model-package encoding of one immutable weight tensor.</summary>
public enum WeightEncoding
{
    /// <summary>IEEE 754 single-precision values.</summary>
    Fp32,
    /// <summary>GGML Q8_0 blocks with 32 values and one FP16 scale per block.</summary>
    GgmlQ8Zero,
}

/// <summary>Bounded byte range inside one model-package file.</summary>
/// <param name="File">Package-relative source file.</param>
/// <param name="Offset">Zero-based byte offset.</param>
/// <param name="Length">Positive encoded byte count.</param>
public sealed record WeightSourceRange(string File, long Offset, long Length);

/// <summary>Stable source and representation metadata for one immutable tensor.</summary>
/// <param name="Id">Tensor identity used by Constant nodes and regions.</param>
/// <param name="Source">Exact encoded source range.</param>
/// <param name="Encoding">Physical tensor encoding in that range.</param>
/// <param name="LogicalShape">Decoded row-major logical shape.</param>
/// <param name="ContentHash">Optional lower-case SHA-256 of the encoded range.</param>
public sealed record WeightDescriptor(
    TensorId Id,
    WeightSourceRange Source,
    WeightEncoding Encoding,
    TensorShape LogicalShape,
    ContentHash? ContentHash = null);
