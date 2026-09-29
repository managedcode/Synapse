using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.GpuKernels;

/// <summary>
/// Maps a brand-neutral <see cref="DenseDecoderLayout"/> onto the GPU descriptor: byte offsets into the mapped file
/// and each matrix's GGML encoding (ADR-012, ADR-021).
/// </summary>
internal static class GpuLayoutMapping
{
    private const ulong NoTensor = ulong.MaxValue;

    public static NativeDecoderLayerOffsets Offsets(IMappedWeights weights, DenseDecoderLayer layer) => new()
    {
        AttentionNorm = Offset(weights, layer.AttentionNorm, WeightEncoding.Fp32),
        Query = Matrix(weights, layer.Query).Offset,
        Key = Matrix(weights, layer.Key).Offset,
        Value = Matrix(weights, layer.Value).Offset,
        QueryBias = Optional(weights, layer.QueryBias),
        KeyBias = Optional(weights, layer.KeyBias),
        ValueBias = Optional(weights, layer.ValueBias),
        QueryNorm = NoTensor,
        KeyNorm = NoTensor,
        AttentionOutput = Matrix(weights, layer.AttentionOutput).Offset,
        FeedForwardNorm = Offset(weights, layer.FeedForwardNorm, WeightEncoding.Fp32),
        Gate = Matrix(weights, layer.Gate).Offset,
        Up = Matrix(weights, layer.Up).Offset,
        Down = Matrix(weights, layer.Down).Offset,
        Encodings = PackEncodings(
            weights, [layer.Query, layer.Key, layer.Value, layer.AttentionOutput, layer.Gate, layer.Up, layer.Down]),
    };

    /// <summary>The matrix encodings in the runtime profile: <c>q8_0</c>, or for mixed files such as Q4_K_M <c>q4_k+q6_k</c> (ADR-021).</summary>
    public static string WeightsLabel(DenseDecoderLayout layout) => string.Join('+', layout.Layers
        .SelectMany(layer => new[] { layer.Query, layer.Key, layer.Value, layer.AttentionOutput, layer.Gate, layer.Up, layer.Down })
        .Append(layout.TokenEmbedding)
        .Append(layout.Output)
        .Select(weight => weight.Encoding switch
        {
            WeightEncoding.GgmlQ4K => "q4_k",
            WeightEncoding.GgmlQ6K => "q6_k",
            WeightEncoding.Fp32 => "f32",
            WeightEncoding.GgmlQ8Zero or _ => "q8_0",
        })
        .Distinct()
        .Order(StringComparer.Ordinal));

    /// <summary>A matrix's offset and GGML type ID: Q8_0 (8), Q4_K (12), or Q6_K (14) (ADR-021).</summary>
    public static (ulong Offset, uint Encoding) Matrix(IMappedWeights weights, DecoderWeight weight) => weight.Encoding switch
    {
        WeightEncoding.GgmlQ8Zero => (Offset(weights, weight, weight.Encoding), 8u),
        WeightEncoding.GgmlQ4K => (Offset(weights, weight, weight.Encoding), 12u),
        WeightEncoding.GgmlQ6K => (Offset(weights, weight, weight.Encoding), 14u),
        WeightEncoding.Fp32 or _ => throw new NotSupportedException(
            $"GPU backends read matrices as Q8_0, Q4_K, or Q6_K; the package stores {weight.Encoding}."),
    };

    /// <summary>Four bits per matrix in Q, K, V, O, gate, up, down order, from the lowest.</summary>
    public static ulong PackEncodings(IMappedWeights weights, DecoderWeight[] matrices)
    {
        var packed = 0UL;
        for (var index = 0; index < matrices.Length; index++)
        {
            packed |= (ulong)Matrix(weights, matrices[index]).Encoding << (4 * index);
        }

        return packed;
    }

    public static ulong Optional(IMappedWeights weights, DecoderWeight? weight) =>
        weight is { } present ? Offset(weights, present, WeightEncoding.Fp32) : NoTensor;

    public static ulong Offset(IMappedWeights weights, DecoderWeight weight, WeightEncoding expected)
    {
        if (!string.Equals(weight.Source.File, weights.SourceFile, StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"GPU backends read one mapped file; '{weight.Source.File}' differs from '{weights.SourceFile}'.");
        }

        return weight.Encoding == expected
            ? checked((ulong)weight.Source.Offset)
            : throw new NotSupportedException(
                $"GPU backends read this tensor as {expected}; the package stores {weight.Encoding}.");
    }
}
