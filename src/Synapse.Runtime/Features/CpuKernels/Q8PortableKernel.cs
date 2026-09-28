using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>Cross-platform <see cref="Vector128{T}"/> kernel for CPUs without a specialized integer dot.</summary>
internal static unsafe class Q8PortableKernel
{
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
        for (var row = 0; row < rowCount; row++)
        {
            var weights = first + ((long)row * rowBytes);
            for (var token = 0; token < tokenCount; token++)
            {
                output[((long)token * outputStride) + row] = Row(
                    weights,
                    blocks,
                    activations.Quants + ((long)token * matrix.Columns),
                    activations.Scales + ((long)token * blocks));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Row(byte* row, int blocks, sbyte* activation, float* scales)
    {
        var sum = Vector128<float>.Zero;
        for (var block = 0; block < blocks; block++)
        {
            var weights = (sbyte*)row + (block * Q8Matrix.BlockBytes) + 2;
            var values = activation + (block * Q8Matrix.BlockElements);
            var dot = Dot16(Vector128.Load(weights), Vector128.Load(values)) +
                Dot16(Vector128.Load(weights + 16), Vector128.Load(values + 16));
            var scale = Fp16.ToSingle(*(ushort*)(row + (block * Q8Matrix.BlockBytes))) * scales[block];
            sum += Vector128.ConvertToSingle(dot) * Vector128.Create(scale);
        }

        return Vector128.Sum(sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Dot16(Vector128<sbyte> weights, Vector128<sbyte> values)
    {
        var (weightLow, weightHigh) = Vector128.Widen(weights);
        var (valueLow, valueHigh) = Vector128.Widen(values);
        var (product0, product1) = Vector128.Widen(weightLow * valueLow);
        var (product2, product3) = Vector128.Widen(weightHigh * valueHigh);
        return product0 + product1 + (product2 + product3);
    }
}
