namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

/// <summary>
/// ggml ternary super-blocks of 256 values (TQ1_0 base-3 packing, TQ2_0 2-bit packing), decoded
/// exactly as ggml's reference <c>dequantize_row_tq*</c> functions.
/// </summary>
internal static class GgufTernaryBlocks
{
    private const int Elements = 256;
    private const int Tq1PackedBytes = 48;
    private const int Tq1HighBytes = 4;
    private static readonly byte[] PowersOfThree = [1, 3, 9, 27, 81, 243];

    public static IEnumerable<(GgufType Type, BlockSourceDecoder Decoder)> Create() =>
    [
        (GgufType.TQ1_0, new BlockSourceDecoder("gguf.tq1_0", Elements, 54, block => BlockFields.FiniteHalves(block, 52), DecodeTq1)),
        (GgufType.TQ2_0, new BlockSourceDecoder("gguf.tq2_0", Elements, 66, block => BlockFields.FiniteHalves(block, 64), DecodeTq2)),
    ];

    private static void DecodeTq1(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 52);
        var output = 0;
        var wide = Tq1PackedBytes - (Tq1PackedBytes % 32);
        output = DecodeBase3(block[..wide], 32, d, y, output);
        output = DecodeBase3(block[wide..Tq1PackedBytes], 16, d, y, output);
        var high = block.Slice(Tq1PackedBytes, Tq1HighBytes);
        for (var power = 0; power < 4; power++)
        {
            for (var j = 0; j < Tq1HighBytes; j++)
            {
                y[output++] = Trit(high[j], power) * d;
            }
        }
    }

    private static int DecodeBase3(ReadOnlySpan<byte> packed, int width, float d, Span<float> y, int output)
    {
        for (var start = 0; start < packed.Length; start += width)
        {
            for (var power = 0; power < 5; power++)
            {
                for (var m = 0; m < width; m++)
                {
                    y[output++] = Trit(packed[start + m], power) * d;
                }
            }
        }

        return output;
    }

    private static float Trit(byte packed, int power)
    {
        var shifted = (byte)(packed * PowersOfThree[power]);
        return ((shifted * 3) >> 8) - 1;
    }

    private static void DecodeTq2(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 64);
        var output = 0;
        for (var j = 0; j < 64; j += 32)
        {
            for (var l = 0; l < 4; l++)
            {
                for (var m = 0; m < 32; m++)
                {
                    y[output++] = (((block[j + m] >> (l * 2)) & 3) - 1) * d;
                }
            }
        }
    }
}
