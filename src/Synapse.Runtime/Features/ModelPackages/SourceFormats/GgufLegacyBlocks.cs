using System.Buffers.Binary;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

/// <summary>
/// ggml blocks of 32 values (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4), decoded exactly as
/// ggml's reference <c>dequantize_row_*</c> functions. Low nibbles hold the first 16 values
/// of a block and high nibbles the last 16.
/// </summary>
internal static class GgufLegacyBlocks
{
    private const int Elements = 32;
    private const int Half = Elements / 2;

    private static readonly sbyte[] NonLinear4 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    private static readonly sbyte[] Fp4E2M1Doubled = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    private const int MaximumFp4Magnitude = 12;

    public static IEnumerable<(GgufType Type, BlockSourceDecoder Decoder)> Create() =>
    [
        (GgufType.Q4_0, Block("q4_0", 18, block => BlockFields.FiniteHalves(block, 0), DecodeQ4_0)),
        (GgufType.Q4_1, Block("q4_1", 20, block => BlockFields.FiniteHalves(block, 0, 2), DecodeQ4_1)),
        (GgufType.Q5_0, Block("q5_0", 22, block => BlockFields.FiniteHalves(block, 0), DecodeQ5_0)),
        (GgufType.Q5_1, Block("q5_1", 24, block => BlockFields.FiniteHalves(block, 0, 2), DecodeQ5_1)),
        (GgufType.Q8_0, Block("q8_0", 34, block => BlockFields.FiniteHalves(block, 0), DecodeQ8_0)),
        (GgufType.IQ4_NL, Block("iq4_nl", 18, block => BlockFields.FiniteHalves(block, 0), DecodeIq4Nl)),
        (GgufType.MXFP4, Block("mxfp4", 17, ValidateMxfp4, DecodeMxfp4)),
    ];

    private static BlockSourceDecoder Block(string name, int bytes, BlockValidator validate, BlockDecoder decode) =>
        new($"gguf.{name}", Elements, bytes, validate, decode);

    private static void DecodeQ4_0(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var qs = block[2..];
        for (var j = 0; j < Half; j++)
        {
            y[j] = ((qs[j] & 0x0F) - 8) * d;
            y[j + Half] = ((qs[j] >> 4) - 8) * d;
        }
    }

    private static void DecodeQ4_1(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var m = BlockFields.Half(block, 2);
        var qs = block[4..];
        for (var j = 0; j < Half; j++)
        {
            y[j] = ((qs[j] & 0x0F) * d) + m;
            y[j + Half] = ((qs[j] >> 4) * d) + m;
        }
    }

    private static void DecodeQ5_0(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var qh = BinaryPrimitives.ReadUInt32LittleEndian(block[2..]);
        var qs = block[6..];
        for (var j = 0; j < Half; j++)
        {
            var high0 = (int)((qh >> j) << 4) & 0x10;
            var high1 = (int)(qh >> (j + 12)) & 0x10;
            y[j] = (((qs[j] & 0x0F) | high0) - 16) * d;
            y[j + Half] = (((qs[j] >> 4) | high1) - 16) * d;
        }
    }

    private static void DecodeQ5_1(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var m = BlockFields.Half(block, 2);
        var qh = BinaryPrimitives.ReadUInt32LittleEndian(block[4..]);
        var qs = block[8..];
        for (var j = 0; j < Half; j++)
        {
            var high0 = (int)((qh >> j) << 4) & 0x10;
            var high1 = (int)(qh >> (j + 12)) & 0x10;
            y[j] = (((qs[j] & 0x0F) | high0) * d) + m;
            y[j + Half] = (((qs[j] >> 4) | high1) * d) + m;
        }
    }

    private static void DecodeQ8_0(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        for (var j = 0; j < Elements; j++)
        {
            y[j] = unchecked((sbyte)block[2 + j]) * d;
        }
    }

    private static void DecodeIq4Nl(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var qs = block[2..];
        for (var j = 0; j < Half; j++)
        {
            y[j] = d * NonLinear4[qs[j] & 0x0F];
            y[j + Half] = d * NonLinear4[qs[j] >> 4];
        }
    }

    private static SourceEncodingFailure? ValidateMxfp4(ReadOnlySpan<byte> block) =>
        float.IsFinite(E8M0Half(block[0]) * MaximumFp4Magnitude) ? null : SourceEncodingFailure.InvalidScale;

    private static void DecodeMxfp4(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = E8M0Half(block[0]);
        var qs = block[1..];
        for (var j = 0; j < Half; j++)
        {
            y[j] = Fp4E2M1Doubled[qs[j] & 0x0F] * d;
            y[j + Half] = Fp4E2M1Doubled[qs[j] >> 4] * d;
        }
    }

    /// <summary>Half of the E8M0 scale <c>2^(e-127)</c>, matching ggml's doubled FP4 table.</summary>
    private static float E8M0Half(byte exponent)
    {
        var bits = exponent < 2 ? 0x0020_0000u << exponent : (uint)(exponent - 1) << 23;
        return BitConverter.UInt32BitsToSingle(bits);
    }
}
