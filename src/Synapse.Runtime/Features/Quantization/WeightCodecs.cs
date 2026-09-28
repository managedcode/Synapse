using System.Globalization;
using System.Text.RegularExpressions;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>A grouped row-major weight encoding with a scalar reference implementation.</summary>
public interface IWeightCodec
{
    /// <summary>Stable encoding identity used in profiles and cache keys.</summary>
    string EncodingId { get; }

    /// <summary>Exact stored byte length of a matrix.</summary>
    long GetEncodedLength(int rows, int columns);

    /// <summary>Quantizes a row-major FP32 matrix.</summary>
    void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination);

    /// <summary>Dequantizes a stored matrix.</summary>
    void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination);

    /// <summary>Computes <c>output = W input</c> from the stored matrix.</summary>
    void Multiply(ReadOnlySpan<byte> encoded, int rows, int columns, ReadOnlySpan<float> input, Span<float> output);
}

/// <summary>Registered on-the-fly weight codecs, resolvable from a profile's encoding identity.</summary>
public static partial class WeightCodecs
{
    /// <summary>The spec §6.1 <c>syn.q4.symmetric.g64.v1</c> codec.</summary>
    public static IWeightCodec Q4 => SynQ4BlockCodec.Codec;

    /// <summary>The <c>syn.ternary.absmean.g64.v1</c> codec.</summary>
    public static IWeightCodec Ternary { get; } = new TernaryCodec();

    /// <summary>Creates a symmetric codec of any supported width (2..8 bits) and group size.</summary>
    public static IWeightCodec Symmetric(int bits, int groupElements) => SymmetricGroupCodec.Create(bits, groupElements);

    /// <summary>Resolves a codec from its stable encoding identity.</summary>
    public static bool TryGet(string encodingId, out IWeightCodec? codec)
    {
        codec = null;
        if (string.Equals(encodingId, TernaryBlockCodec.EncodingId, StringComparison.Ordinal))
        {
            codec = Ternary;
            return true;
        }

        var match = SymmetricIdPattern().Match(encodingId ?? string.Empty);
        if (!match.Success ||
            !int.TryParse(match.Groups["bits"].ValueSpan, NumberStyles.None, CultureInfo.InvariantCulture, out var bits) ||
            !int.TryParse(match.Groups["group"].ValueSpan, NumberStyles.None, CultureInfo.InvariantCulture, out var group) ||
            bits is < SymmetricGroupCodec.MinimumBits or > SymmetricGroupCodec.MaximumBits ||
            group is < 8 or > SymmetricGroupCodec.MaximumGroupElements || group % 8 != 0)
        {
            return false;
        }

        codec = SymmetricGroupCodec.Create(bits, group);
        return true;
    }

    /// <summary>Encodes and immediately decodes a matrix to obtain its effective weights.</summary>
    public static float[] RoundTrip(this IWeightCodec codec, ReadOnlySpan<float> weights, int rows, int columns)
    {
        ArgumentNullException.ThrowIfNull(codec);
        var encoded = new byte[checked((int)codec.GetEncodedLength(rows, columns))];
        var decoded = new float[weights.Length];
        codec.Encode(weights, rows, columns, encoded);
        codec.Decode(encoded, rows, columns, decoded);
        return decoded;
    }

    [GeneratedRegex(@"^syn\.q(?<bits>[0-9])\.symmetric\.g(?<group>[0-9]{1,4})\.v1$", RegexOptions.CultureInvariant)]
    private static partial Regex SymmetricIdPattern();

    private sealed class TernaryCodec : IWeightCodec
    {
        public string EncodingId => TernaryBlockCodec.EncodingId;

        public long GetEncodedLength(int rows, int columns) => TernaryBlockCodec.GetEncodedLength(rows, columns);

        public void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination) =>
            TernaryBlockCodec.Encode(weights, rows, columns, destination);

        public void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination) =>
            TernaryBlockCodec.Decode(encoded, rows, columns, destination);

        public void Multiply(ReadOnlySpan<byte> encoded, int rows, int columns, ReadOnlySpan<float> input, Span<float> output) =>
            TernaryBlockCodec.Multiply(encoded, rows, columns, input, output);
    }
}
