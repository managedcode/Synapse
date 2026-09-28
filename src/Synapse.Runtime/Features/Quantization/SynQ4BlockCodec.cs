namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>
/// The spec §6.1 encoding <c>syn.q4.symmetric.g64.v1</c>: the 4-bit, 64-element member
/// of <see cref="SymmetricGroupCodec"/>. Each group stores an FP16 scale and 32 packed
/// bytes (even elements in the low nibble), 4.25 stored bits per weight.
/// This is not GGUF <c>Q4_K</c>.
/// </summary>
public static class SynQ4BlockCodec
{
    /// <summary>Stable encoding identity.</summary>
    public const string EncodingId = "syn.q4.symmetric.g64.v1";

    /// <summary>Weights per group.</summary>
    public const int GroupElements = 64;

    /// <summary>Stored bytes per group: a two-byte scale and 32 packed code bytes.</summary>
    public const int GroupBytes = GroupedRowLayout.ScaleBytes + (GroupElements / 2);

    /// <summary>The codec instance behind this encoding.</summary>
    public static SymmetricGroupCodec Codec { get; } = SymmetricGroupCodec.Create(4, GroupElements);

    /// <summary>Returns the exact stored byte length of a matrix.</summary>
    public static long GetEncodedLength(int rows, int columns) => Codec.GetEncodedLength(rows, columns);

    /// <summary>Returns stored bits per logical weight, including scales and padding.</summary>
    public static double GetStoredBitsPerWeight(int rows, int columns) => Codec.GetStoredBitsPerWeight(rows, columns);

    /// <summary>Quantizes a row-major FP32 matrix after validating every source weight.</summary>
    public static void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination) =>
        Codec.Encode(weights, rows, columns, destination);

    /// <summary>Dequantizes a matrix after validating every stored group.</summary>
    public static void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination) =>
        Codec.Decode(encoded, rows, columns, destination);

    /// <summary>Computes <c>output = W input</c> directly from encoded groups.</summary>
    public static void Multiply(
        ReadOnlySpan<byte> encoded,
        int rows,
        int columns,
        ReadOnlySpan<float> input,
        Span<float> output) => Codec.Multiply(encoded, rows, columns, input, output);
}
