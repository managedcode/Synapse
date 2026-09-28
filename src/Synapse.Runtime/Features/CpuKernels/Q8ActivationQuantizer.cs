using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// GGML-compatible Q8_0 activation quantization: per 32-value block, <c>d = max|x| / 127</c> is stored as
/// FP16 and codes are <c>round_half_even(x * (1 / d))</c> using the unrounded <c>d</c> (ADR-006).
/// </summary>
internal static class Q8ActivationQuantizer
{
    public const int BlockElements = 32;

    public static void QuantizeRow(ReadOnlySpan<float> input, Span<sbyte> quants, Span<float> scales)
    {
        if (input.Length == 0 || input.Length % BlockElements != 0 ||
            quants.Length != input.Length || scales.Length != input.Length / BlockElements)
        {
            throw new ArgumentException(
                $"Q8_0 activation rows need a positive multiple of {BlockElements} values, equal codes, " +
                $"and one scale per block; got {input.Length} values, {quants.Length} codes, {scales.Length} scales.");
        }

        for (var block = 0; block < scales.Length; block++)
        {
            scales[block] = QuantizeBlock(
                input.Slice(block * BlockElements, BlockElements),
                quants.Slice(block * BlockElements, BlockElements));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float QuantizeBlock(ReadOnlySpan<float> values, Span<sbyte> codes)
    {
        ref var source = ref MemoryMarshal.GetReference(values);
        var v0 = Vector128.LoadUnsafe(ref source, 0);
        var v1 = Vector128.LoadUnsafe(ref source, 4);
        var v2 = Vector128.LoadUnsafe(ref source, 8);
        var v3 = Vector128.LoadUnsafe(ref source, 12);
        var v4 = Vector128.LoadUnsafe(ref source, 16);
        var v5 = Vector128.LoadUnsafe(ref source, 20);
        var v6 = Vector128.LoadUnsafe(ref source, 24);
        var v7 = Vector128.LoadUnsafe(ref source, 28);
        var maximum = Vector128.Max(
            Vector128.Max(Vector128.Max(Vector128.Abs(v0), Vector128.Abs(v1)), Vector128.Max(Vector128.Abs(v2), Vector128.Abs(v3))),
            Vector128.Max(Vector128.Max(Vector128.Abs(v4), Vector128.Abs(v5)), Vector128.Max(Vector128.Abs(v6), Vector128.Abs(v7))));
        var absoluteMaximum = MathF.Max(
            MathF.Max(maximum.GetElement(0), maximum.GetElement(1)),
            MathF.Max(maximum.GetElement(2), maximum.GetElement(3)));
        var scale = absoluteMaximum / 127f;
        var inverse = Vector128.Create(scale == 0 ? 0f : 1f / scale);
        ref var destination = ref MemoryMarshal.GetReference(codes);
        Pack(v0, v1, v2, v3, inverse).StoreUnsafe(ref destination, 0);
        Pack(v4, v5, v6, v7, inverse).StoreUnsafe(ref destination, 16);
        return (float)(Half)scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<sbyte> Pack(
        Vector128<float> first,
        Vector128<float> second,
        Vector128<float> third,
        Vector128<float> fourth,
        Vector128<float> inverse)
    {
        var low = Vector128.Narrow(Round(first, inverse), Round(second, inverse));
        var high = Vector128.Narrow(Round(third, inverse), Round(fourth, inverse));
        return Vector128.Narrow(low, high);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Round(Vector128<float> values, Vector128<float> inverse) =>
        Vector128.ConvertToInt32(Vector128.Round(values * inverse));
}
