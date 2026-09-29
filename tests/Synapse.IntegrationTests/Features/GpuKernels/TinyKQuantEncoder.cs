namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>
/// Encodes FP32 values into valid ggml Q4_K and Q6_K super-blocks (256 values) for generated test models. The
/// choice of scales is simple, not optimal; what matters is that every block is exactly decodable by ggml's
/// layout, which the runtime decoders and kernels then have to reproduce.
/// </summary>
internal static class TinyKQuantEncoder
{
    public const int SuperBlock = 256;

    /// <summary>Q4_K: d, dmin, 12 packed 6-bit scale/min bytes, then 128 bytes of 4-bit codes (144 bytes).</summary>
    public static byte[] Q4K(ReadOnlySpan<float> values)
    {
        var blocks = values.Length / SuperBlock;
        var data = new byte[blocks * 144];
        for (var block = 0; block < blocks; block++)
        {
            EncodeQ4KBlock(values.Slice(block * SuperBlock, SuperBlock), data.AsSpan(block * 144, 144));
        }

        return data;
    }

    /// <summary>Q6_K: 128 low-nibble bytes, 64 high-bit bytes, 16 signed scales, then d (210 bytes).</summary>
    public static byte[] Q6K(ReadOnlySpan<float> values)
    {
        var blocks = values.Length / SuperBlock;
        var data = new byte[blocks * 210];
        for (var block = 0; block < blocks; block++)
        {
            EncodeQ6KBlock(values.Slice(block * SuperBlock, SuperBlock), data.AsSpan(block * 210, 210));
        }

        return data;
    }

    private static void EncodeQ4KBlock(ReadOnlySpan<float> x, Span<byte> block)
    {
        Span<float> steps = stackalloc float[8];
        Span<float> offsets = stackalloc float[8];
        for (var sub = 0; sub < 8; sub++)
        {
            var slice = x.Slice(sub * 32, 32);
            var low = MathF.Min(0, Min(slice));
            steps[sub] = (Max(slice) - low) / 15f;
            offsets[sub] = -low;
        }

        var d = (float)(Half)(Max(steps) / 63f);
        var dmin = (float)(Half)(Max(offsets) / 63f);
        Span<int> scale = stackalloc int[8];
        Span<int> minimum = stackalloc int[8];
        for (var sub = 0; sub < 8; sub++)
        {
            scale[sub] = d == 0 ? 0 : Math.Clamp((int)MathF.Round(steps[sub] / d), 0, 63);
            minimum[sub] = dmin == 0 ? 0 : Math.Clamp((int)MathF.Round(offsets[sub] / dmin), 0, 63);
        }

        BitConverter.TryWriteBytes(block, (Half)d);
        BitConverter.TryWriteBytes(block[2..], (Half)dmin);
        var packed = block.Slice(4, 12);
        for (var k = 0; k < 4; k++)
        {
            packed[k] = (byte)((scale[k] & 63) | ((scale[k + 4] >> 4) << 6));
            packed[k + 4] = (byte)((minimum[k] & 63) | ((minimum[k + 4] >> 4) << 6));
            packed[k + 8] = (byte)((scale[k + 4] & 0xF) | ((minimum[k + 4] & 0xF) << 4));
        }

        var codes = block.Slice(16, 128);
        for (var sub = 0; sub < 8; sub++)
        {
            var step = d * scale[sub];
            var bias = dmin * minimum[sub];
            for (var l = 0; l < 32; l++)
            {
                var q = step == 0 ? 0 : Math.Clamp((int)MathF.Round((x[(sub * 32) + l] + bias) / step), 0, 15);
                codes[(sub / 2 * 32) + l] |= (byte)(sub % 2 == 0 ? q : q << 4);
            }
        }
    }

    private static void EncodeQ6KBlock(ReadOnlySpan<float> x, Span<byte> block)
    {
        Span<float> steps = stackalloc float[16];
        for (var group = 0; group < 16; group++)
        {
            steps[group] = MaxAbs(x.Slice(group * 16, 16)) / 31f;
        }

        var d = (float)(Half)(Max(steps) / 127f);
        BitConverter.TryWriteBytes(block[208..], (Half)d);
        for (var group = 0; group < 16; group++)
        {
            block[192 + group] = (byte)(d == 0 ? 1 : Math.Clamp((int)MathF.Round(steps[group] / d), 1, 127));
        }

        for (var index = 0; index < SuperBlock; index++)
        {
            // Value i sits in half h, quarter r/32, lane l. ggml's scale index h*8 + l/16 + 2*quarter equals i/16.
            var (half, rest) = Math.DivRem(index, 128);
            var (quarter, l) = Math.DivRem(rest, 32);
            var step = d * (sbyte)block[192 + (index / 16)];
            var q = (step == 0 ? 0 : Math.Clamp((int)MathF.Round(x[index] / step), -32, 31)) + 32;
            var low = (half * 64) + (quarter % 2 == 0 ? l : l + 32);
            block[low] |= (byte)(quarter < 2 ? q & 0xF : (q & 0xF) << 4);
            block[128 + (half * 32) + l] |= (byte)(((q >> 4) & 3) << (2 * quarter));
        }
    }

    private static float Min(ReadOnlySpan<float> values)
    {
        var result = float.PositiveInfinity;
        foreach (var value in values)
        {
            result = MathF.Min(result, value);
        }

        return result;
    }

    private static float Max(ReadOnlySpan<float> values)
    {
        var result = float.NegativeInfinity;
        foreach (var value in values)
        {
            result = MathF.Max(result, value);
        }

        return result;
    }

    private static float MaxAbs(ReadOnlySpan<float> values)
    {
        var result = 0f;
        foreach (var value in values)
        {
            result = MathF.Max(result, MathF.Abs(value));
        }

        return result;
    }
}
