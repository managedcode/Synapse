using System.Buffers.Binary;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

/// <summary>
/// ggml k-quant super-blocks of 256 values (Q2_K..Q6_K, Q8_K, IQ4_XS), decoded exactly as
/// ggml's reference <c>dequantize_row_*</c> functions, including their evaluation order.
/// </summary>
internal static class GgufKQuantBlocks
{
    private const int Elements = 256;

    private static readonly sbyte[] NonLinear4 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    private const float MaximumQ8Magnitude = 128f;

    public static IEnumerable<(GgufType Type, BlockSourceDecoder Decoder)> Create() =>
    [
        (GgufType.Q2_K, Block("q2_K", 84, block => BlockFields.FiniteHalves(block, 80, 82), DecodeQ2K)),
        (GgufType.Q3_K, Block("q3_K", 110, block => BlockFields.FiniteHalves(block, 108), DecodeQ3K)),
        (GgufType.Q4_K, Block("q4_K", 144, block => BlockFields.FiniteHalves(block, 0, 2), DecodeQ4K)),
        (GgufType.Q5_K, Block("q5_K", 176, block => BlockFields.FiniteHalves(block, 0, 2), DecodeQ5K)),
        (GgufType.Q6_K, Block("q6_K", 210, block => BlockFields.FiniteHalves(block, 208), DecodeQ6K)),
        (GgufType.Q8_K, Block("q8_K", 292, ValidateQ8K, DecodeQ8K)),
        (GgufType.IQ4_XS, Block("iq4_xs", 136, block => BlockFields.FiniteHalves(block, 0), DecodeIq4Xs)),
    ];

    private static BlockSourceDecoder Block(string name, int bytes, BlockValidator validate, BlockDecoder decode) =>
        new($"gguf.{name}", Elements, bytes, validate, decode);

    private static void DecodeQ2K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var scales = block[..16];
        var d = BlockFields.Half(block, 80);
        var min = BlockFields.Half(block, 82);
        var output = 0;
        var scale = 0;
        for (var half = 0; half < 2; half++)
        {
            var q = block.Slice(16 + (half * 32), 32);
            for (var shift = 0; shift < 8; shift += 2)
            {
                for (var part = 0; part < 2; part++)
                {
                    var packed = scales[scale++];
                    var dl = d * (packed & 0x0F);
                    var ml = min * (packed >> 4);
                    for (var l = 0; l < 16; l++)
                    {
                        y[output++] = (dl * ((q[(part * 16) + l] >> shift) & 3)) - ml;
                    }
                }
            }
        }
    }

    private static void DecodeQ3K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var hmask = block[..32];
        var d = BlockFields.Half(block, 108);
        Span<byte> scales = stackalloc byte[16];
        UnpackQ3Scales(block.Slice(96, 12), scales);
        var output = 0;
        var scale = 0;
        var mask = 1;
        for (var half = 0; half < 2; half++)
        {
            var q = block.Slice(32 + (half * 32), 32);
            for (var shift = 0; shift < 8; shift += 2)
            {
                for (var part = 0; part < 2; part++)
                {
                    var dl = d * (unchecked((sbyte)scales[scale++]) - 32);
                    for (var l = 0; l < 16; l++)
                    {
                        var index = (part * 16) + l;
                        var low = (q[index] >> shift) & 3;
                        y[output++] = dl * (low - ((hmask[index] & mask) != 0 ? 0 : 4));
                    }
                }

                mask <<= 1;
            }
        }
    }

    private static void UnpackQ3Scales(ReadOnlySpan<byte> packed, Span<byte> scales)
    {
        const uint LowMask = 0x0303_0303;
        const uint NibbleMask = 0x0F0F_0F0F;
        var a0 = BinaryPrimitives.ReadUInt32LittleEndian(packed);
        var a1 = BinaryPrimitives.ReadUInt32LittleEndian(packed[4..]);
        var tmp = BinaryPrimitives.ReadUInt32LittleEndian(packed[8..]);
        BinaryPrimitives.WriteUInt32LittleEndian(scales, (a0 & NibbleMask) | ((tmp & LowMask) << 4));
        BinaryPrimitives.WriteUInt32LittleEndian(scales[4..], (a1 & NibbleMask) | (((tmp >> 2) & LowMask) << 4));
        BinaryPrimitives.WriteUInt32LittleEndian(scales[8..], ((a0 >> 4) & NibbleMask) | (((tmp >> 4) & LowMask) << 4));
        BinaryPrimitives.WriteUInt32LittleEndian(scales[12..], ((a1 >> 4) & NibbleMask) | (((tmp >> 6) & LowMask) << 4));
    }

    private static void DecodeQ4K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var min = BlockFields.Half(block, 2);
        var scales = block.Slice(4, 12);
        var output = 0;
        for (var chunk = 0; chunk < 4; chunk++)
        {
            var q = block.Slice(16 + (chunk * 32), 32);
            var (scale1, min1) = ScaleMinK4(chunk * 2, scales);
            var (scale2, min2) = ScaleMinK4((chunk * 2) + 1, scales);
            float d1 = d * scale1, m1 = min * min1, d2 = d * scale2, m2 = min * min2;
            for (var l = 0; l < 32; l++)
            {
                y[output++] = (d1 * (q[l] & 0x0F)) - m1;
            }

            for (var l = 0; l < 32; l++)
            {
                y[output++] = (d2 * (q[l] >> 4)) - m2;
            }
        }
    }

    private static void DecodeQ5K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var min = BlockFields.Half(block, 2);
        var scales = block.Slice(4, 12);
        var qh = block.Slice(16, 32);
        var output = 0;
        for (var chunk = 0; chunk < 4; chunk++)
        {
            var ql = block.Slice(48 + (chunk * 32), 32);
            var (scale1, min1) = ScaleMinK4(chunk * 2, scales);
            var (scale2, min2) = ScaleMinK4((chunk * 2) + 1, scales);
            float d1 = d * scale1, m1 = min * min1, d2 = d * scale2, m2 = min * min2;
            int u1 = 1 << (chunk * 2), u2 = 2 << (chunk * 2);
            for (var l = 0; l < 32; l++)
            {
                y[output++] = (d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0))) - m1;
            }

            for (var l = 0; l < 32; l++)
            {
                y[output++] = (d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0))) - m2;
            }
        }
    }

    private static (int Scale, int Min) ScaleMinK4(int index, ReadOnlySpan<byte> scales) =>
        index < 4
            ? (scales[index] & 63, scales[index + 4] & 63)
            : ((scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));

    private static void DecodeQ6K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 208);
        for (var half = 0; half < 2; half++)
        {
            var ql = block.Slice(half * 64, 64);
            var qh = block.Slice(128 + (half * 32), 32);
            var sc = block.Slice(192 + (half * 8), 8);
            var target = y.Slice(half * 128, 128);
            for (var l = 0; l < 32; l++)
            {
                var group = l / 16;
                var q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                var q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                var q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                var q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                target[l] = d * unchecked((sbyte)sc[group]) * q1;
                target[l + 32] = d * unchecked((sbyte)sc[group + 2]) * q2;
                target[l + 64] = d * unchecked((sbyte)sc[group + 4]) * q3;
                target[l + 96] = d * unchecked((sbyte)sc[group + 6]) * q4;
            }
        }
    }

    private static SourceEncodingFailure? ValidateQ8K(ReadOnlySpan<byte> block) =>
        float.IsFinite(BinaryPrimitives.ReadSingleLittleEndian(block) * MaximumQ8Magnitude) ? null : SourceEncodingFailure.InvalidScale;

    private static void DecodeQ8K(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BinaryPrimitives.ReadSingleLittleEndian(block);
        for (var j = 0; j < Elements; j++)
        {
            y[j] = d * unchecked((sbyte)block[4 + j]);
        }
    }

    private static void DecodeIq4Xs(ReadOnlySpan<byte> block, Span<float> y)
    {
        var d = BlockFields.Half(block, 0);
        var scalesHigh = BinaryPrimitives.ReadUInt16LittleEndian(block[2..]);
        var scalesLow = block.Slice(4, 4);
        for (var sub = 0; sub < 8; sub++)
        {
            var qs = block.Slice(8 + (sub * 16), 16);
            var ls = ((scalesLow[sub / 2] >> (4 * (sub % 2))) & 0x0F) | (((scalesHigh >> (2 * sub)) & 3) << 4);
            var dl = d * (ls - 32);
            var target = y.Slice(sub * 32, 32);
            for (var j = 0; j < 16; j++)
            {
                target[j] = dl * NonLinear4[qs[j] & 0x0F];
                target[j + 16] = dl * NonLinear4[qs[j] >> 4];
            }
        }
    }
}
