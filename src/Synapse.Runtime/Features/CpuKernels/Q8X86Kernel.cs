using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>x64 AVX2 kernel with an optional AVX-VNNI integer dot; four weight rows share each activation load.</summary>
internal static unsafe class Q8X86Kernel
{
    public static bool IsAvx2Supported => Avx2.IsSupported && Fma.IsSupported;

    public static bool IsVnniSupported => IsAvx2Supported && AvxVnni.IsSupported;

    public static void Multiply(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride,
        bool useVnni)
    {
        if (useVnni)
        {
            Multiply<VnniDot>(matrix, rowStart, rowCount, activations, tokenCount, output, outputStride);
        }
        else
        {
            Multiply<Avx2Dot>(matrix, rowStart, rowCount, activations, tokenCount, output, outputStride);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Multiply<TDot>(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride)
        where TDot : struct, IX86Dot
    {
        var rowBytes = matrix.RowBytes;
        var blocks = matrix.BlocksPerRow;
        var first = matrix.Data + ((long)rowStart * rowBytes);
        var quadRows = blocks >= Q8Matrix.LongRowBlocks ? 0 : rowCount & ~3;
        for (var row = 0; row < quadRows; row += 4)
        {
            var weights = first + ((long)row * rowBytes);
            for (var token = 0; token < tokenCount; token++)
            {
                Rows4<TDot>(
                    weights,
                    rowBytes,
                    blocks,
                    activations.Quants + ((long)token * matrix.Columns),
                    activations.Scales + ((long)token * blocks)).Store(output + ((long)token * outputStride) + row);
            }
        }

        for (var row = quadRows; row < rowCount; row++)
        {
            var weights = first + ((long)row * rowBytes);
            for (var token = 0; token < tokenCount; token++)
            {
                output[((long)token * outputStride) + row] = Vector256.Sum(Row<TDot>(
                    weights,
                    blocks,
                    activations.Quants + ((long)token * matrix.Columns),
                    activations.Scales + ((long)token * blocks)));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Rows4<TDot>(byte* row0, int rowBytes, int blocks, sbyte* activation, float* scales)
        where TDot : struct, IX86Dot
    {
        var row1 = row0 + rowBytes;
        var row2 = row1 + rowBytes;
        var row3 = row2 + rowBytes;
        var sum0 = Vector256<float>.Zero;
        var sum1 = Vector256<float>.Zero;
        var sum2 = Vector256<float>.Zero;
        var sum3 = Vector256<float>.Zero;
        for (var block = 0; block < blocks; block++)
        {
            var offset = block * Q8Matrix.BlockBytes;
            var values = Vector256.Load(activation + (block * Q8Matrix.BlockElements));
            var scale = scales[block];
            sum0 = Accumulate<TDot>(sum0, row0 + offset, values, scale);
            sum1 = Accumulate<TDot>(sum1, row1 + offset, values, scale);
            sum2 = Accumulate<TDot>(sum2, row2 + offset, values, scale);
            sum3 = Accumulate<TDot>(sum3, row3 + offset, values, scale);
        }

        return Vector128.Create(Vector256.Sum(sum0), Vector256.Sum(sum1), Vector256.Sum(sum2), Vector256.Sum(sum3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Row<TDot>(byte* row, int blocks, sbyte* activation, float* scales)
        where TDot : struct, IX86Dot
    {
        var sum0 = Vector256<float>.Zero;
        var sum1 = Vector256<float>.Zero;
        var block = 0;
        for (; block + 1 < blocks; block += 2)
        {
            var weights = row + (block * Q8Matrix.BlockBytes);
            var values = activation + (block * Q8Matrix.BlockElements);
            sum0 = Accumulate<TDot>(sum0, weights, Vector256.Load(values), scales[block]);
            sum1 = Accumulate<TDot>(sum1, weights + Q8Matrix.BlockBytes, Vector256.Load(values + 32), scales[block + 1]);
        }

        for (; block < blocks; block++)
        {
            var values = Vector256.Load(activation + (block * Q8Matrix.BlockElements));
            sum0 = Accumulate<TDot>(sum0, row + (block * Q8Matrix.BlockBytes), values, scales[block]);
        }

        return sum0 + sum1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Accumulate<TDot>(
        Vector256<float> sum,
        byte* block,
        Vector256<sbyte> activation,
        float activationScale)
        where TDot : struct, IX86Dot
    {
        var weights = Vector256.Load((sbyte*)block + 2);
        var dot = TDot.Dot(Avx2.Abs(weights), Avx2.Sign(activation, weights));
        var scale = Vector256.Create(Fp16.ToSingle(*(ushort*)block) * activationScale);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(dot), scale, sum);
    }

    private interface IX86Dot
    {
        static abstract Vector256<int> Dot(Vector256<byte> magnitudes, Vector256<sbyte> signedValues);
    }

    private readonly struct Avx2Dot : IX86Dot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<byte> magnitudes, Vector256<sbyte> signedValues) =>
            Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(magnitudes, signedValues), Vector256.Create((short)1));
    }

    private readonly struct VnniDot : IX86Dot
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<int> Dot(Vector256<byte> magnitudes, Vector256<sbyte> signedValues) =>
            AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, magnitudes, signedValues);
    }
}
