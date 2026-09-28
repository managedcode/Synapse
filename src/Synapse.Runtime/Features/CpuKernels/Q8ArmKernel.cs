using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// ARM64 <c>sdot</c> kernel. Short rows run four at a time so each activation load serves four rows; long
/// rows run as one sequential stream with four independent accumulators, which measured faster on
/// feed-forward down projections.
/// </summary>
internal static unsafe class Q8ArmKernel
{
    public static bool IsSupported => Dp.IsSupported && AdvSimd.Arm64.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Multiply(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride)
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
                var sums = Rows4(
                    weights,
                    rowBytes,
                    blocks,
                    activations.Quants + ((long)token * matrix.Columns),
                    activations.Scales + ((long)token * blocks));
                sums.Store(output + ((long)token * outputStride) + row);
            }
        }

        for (var row = quadRows; row < rowCount; row++)
        {
            var weights = first + ((long)row * rowBytes);
            for (var token = 0; token < tokenCount; token++)
            {
                output[((long)token * outputStride) + row] = Row1(
                    weights,
                    blocks,
                    activations.Quants + ((long)token * matrix.Columns),
                    activations.Scales + ((long)token * blocks));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Rows4(byte* row0, int rowBytes, int blocks, sbyte* activation, float* scales)
    {
        var row1 = row0 + rowBytes;
        var row2 = row1 + rowBytes;
        var row3 = row2 + rowBytes;
        var sum = Vector128<float>.Zero;
        for (var block = 0; block < blocks; block++)
        {
            var offset = block * Q8Matrix.BlockBytes;
            var low = Vector128.Load(activation);
            var high = Vector128.Load(activation + 16);
            activation += Q8Matrix.BlockElements;
            var dots = AdvSimd.Arm64.AddPairwise(
                AdvSimd.Arm64.AddPairwise(Dot(row0 + offset, low, high), Dot(row1 + offset, low, high)),
                AdvSimd.Arm64.AddPairwise(Dot(row2 + offset, low, high), Dot(row3 + offset, low, high)));
            var weightScales = Fp16.ToSingle(
                *(ushort*)(row0 + offset),
                *(ushort*)(row1 + offset),
                *(ushort*)(row2 + offset),
                *(ushort*)(row3 + offset));
            sum = AdvSimd.FusedMultiplyAdd(
                sum,
                AdvSimd.ConvertToSingle(dots),
                weightScales * Vector128.Create(scales[block]));
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Row1(byte* row, int blocks, sbyte* activation, float* scales)
    {
        var sum0 = Vector128<float>.Zero;
        var sum1 = Vector128<float>.Zero;
        var sum2 = Vector128<float>.Zero;
        var sum3 = Vector128<float>.Zero;
        var block = 0;
        for (; block + 3 < blocks; block += 4)
        {
            var weights = row + (block * Q8Matrix.BlockBytes);
            var values = activation + (block * Q8Matrix.BlockElements);
            sum0 = Accumulate(sum0, weights, values, scales[block]);
            sum1 = Accumulate(sum1, weights + Q8Matrix.BlockBytes, values + 32, scales[block + 1]);
            sum2 = Accumulate(sum2, weights + (2 * Q8Matrix.BlockBytes), values + 64, scales[block + 2]);
            sum3 = Accumulate(sum3, weights + (3 * Q8Matrix.BlockBytes), values + 96, scales[block + 3]);
        }

        for (; block < blocks; block++)
        {
            sum0 = Accumulate(
                sum0,
                row + (block * Q8Matrix.BlockBytes),
                activation + (block * Q8Matrix.BlockElements),
                scales[block]);
        }

        return Vector128.Sum(sum0 + sum1 + (sum2 + sum3));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Accumulate(Vector128<float> sum, byte* block, sbyte* values, float activationScale)
    {
        var dot = Dot(block, Vector128.Load(values), Vector128.Load(values + 16));
        var scale = Fp16.ToSingle(*(ushort*)block) * activationScale;
        return AdvSimd.FusedMultiplyAdd(sum, AdvSimd.ConvertToSingle(dot), Vector128.Create(scale));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Dot(byte* block, Vector128<sbyte> low, Vector128<sbyte> high) =>
        Dp.DotProduct(
            Dp.DotProduct(Vector128<int>.Zero, Vector128.Load((sbyte*)block + 2), low),
            Vector128.Load((sbyte*)block + 18),
            high);
}
