using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

internal static unsafe class Q8KernelOracle
{
    public const double RelativeTolerance = 2e-5;

    public static Q8ActivationBuffer CreateActivations(int columns, int tokens, int seed)
    {
        var random = new Random(seed);
        var buffer = new Q8ActivationBuffer(columns, tokens);
        var row = new float[columns];
        for (var token = 0; token < tokens; token++)
        {
            for (var index = 0; index < row.Length; index++)
            {
                row[index] = (float)((random.NextDouble() * 6) - 3);
            }

            buffer.Quantize(token, row);
        }

        return buffer;
    }

    public static float[] Run(
        Q8MatrixKernel kernel,
        Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokens)
    {
        var output = new float[rowCount * tokens];
        fixed (float* pointer = output)
        {
            kernel.Multiply(matrix, rowStart, rowCount, activations, tokens, pointer, rowCount);
        }

        return output;
    }

    /// <summary>Returns the largest error divided by the FP32 accumulation bound for any output.</summary>
    public static double WorstRelativeError(
        Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokens,
        float[] actual)
    {
        var worst = 0.0;
        for (var token = 0; token < tokens; token++)
        {
            for (var row = 0; row < rowCount; row++)
            {
                var (exact, magnitude) = Exact(matrix, rowStart + row, activations, token);
                var error = Math.Abs(actual[(token * rowCount) + row] - exact);
                worst = Math.Max(worst, error / Math.Max(magnitude * RelativeTolerance, 1e-12));
            }
        }

        return worst;
    }

    private static (double Value, double Magnitude) Exact(
        Q8Matrix matrix,
        int row,
        Q8ActivationBuffer activations,
        int token)
    {
        var value = 0.0;
        var magnitude = 0.0;
        var weights = matrix.Data + ((long)row * matrix.RowBytes);
        var quants = activations.GetQuants(token);
        var scales = activations.GetScales(token);
        for (var block = 0; block < matrix.BlocksPerRow; block++)
        {
            var blockPointer = weights + (block * Q8Matrix.BlockBytes);
            var weightScale = (double)BitConverter.UInt16BitsToHalf(*(ushort*)blockPointer);
            long integerDot = 0;
            for (var index = 0; index < Q8Matrix.BlockElements; index++)
            {
                integerDot += ((sbyte*)blockPointer)[2 + index] * quants[(block * Q8Matrix.BlockElements) + index];
            }

            var term = weightScale * scales[block] * integerDot;
            value += term;
            magnitude += Math.Abs(term);
        }

        return (value, magnitude);
    }
}
