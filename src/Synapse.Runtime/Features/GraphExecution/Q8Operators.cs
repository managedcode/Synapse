using System.Runtime.CompilerServices;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution;

internal static unsafe class Q8Operators
{
    private const int BlockElements = 32;
    private const int BlockBytes = 34;
    private const uint Q8Type = 8;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ReadRow(
        GgufFile file,
        GgufTensorInfo tensor,
        int row,
        Span<float> destination)
    {
        if (tensor.Type != Q8Type)
        {
            DecodedRowOperators.ReadRow(file, tensor, row, destination);
            return;
        }

        var columns = GetMatrixColumns(tensor);
        var rows = GetMatrixRows(tensor);
        if ((uint)row >= (uint)rows || destination.Length != columns)
        {
            throw new ArgumentOutOfRangeException(nameof(row));
        }

        var blocksPerRow = columns / BlockElements;
        var rowPointer = file.GetTensorPointer(tensor) + checked(row * blocksPerRow * BlockBytes);
        for (var block = 0; block < blocksPerRow; block++)
        {
            var blockPointer = rowPointer + (block * BlockBytes);
            var scale = ReadScale(blockPointer);
            for (var index = 0; index < BlockElements; index++)
            {
                destination[(block * BlockElements) + index] =
                    scale * unchecked((sbyte)blockPointer[sizeof(ushort) + index]);
            }
        }
    }

    public static void Multiply(
        GgufFile file,
        GgufTensorInfo tensor,
        float[] input,
        float[] output,
        ParallelOptions parallelOptions)
    {
        if (tensor.Type != Q8Type)
        {
            DecodedRowOperators.Multiply(file, tensor, input, output, parallelOptions);
            return;
        }

        var columns = GetMatrixColumns(tensor);
        var rows = GetMatrixRows(tensor);
        if (input.Length != columns || output.Length != rows)
        {
            throw new ArgumentException(
                $"Tensor '{tensor.Name}' expects [{columns}] -> [{rows}], got [{input.Length}] -> [{output.Length}].");
        }

        var blocksPerRow = columns / BlockElements;
        var rowBytes = blocksPerRow * BlockBytes;
        var tensorAddress = (nint)file.GetTensorPointer(tensor);
        fixed (float* inputPointer = input)
        fixed (float* outputPointer = output)
        {
            var inputAddress = (nint)inputPointer;
            var outputAddress = (nint)outputPointer;
            _ = Parallel.For(0, rows, parallelOptions, row =>
            {
                var weights = (byte*)tensorAddress + checked(row * rowBytes);
                var values = (float*)inputAddress;
                var sum = 0.0f;
                for (var block = 0; block < blocksPerRow; block++)
                {
                    var blockPointer = weights + (block * BlockBytes);
                    var scale = ReadScale(blockPointer);
                    var inputOffset = block * BlockElements;
                    var blockSum = 0.0f;
                    for (var index = 0; index < BlockElements; index++)
                    {
                        blockSum += unchecked((sbyte)blockPointer[sizeof(ushort) + index])
                            * values[inputOffset + index];
                    }

                    sum += scale * blockSum;
                }

                ((float*)outputAddress)[row] = sum;
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights,
        float epsilon,
        Span<float> output)
    {
        if (input.Length != weights.Length || output.Length != input.Length)
        {
            throw new ArgumentException("RMSNorm buffers must have identical lengths.");
        }

        var sumOfSquares = 0.0;
        for (var index = 0; index < input.Length; index++)
        {
            sumOfSquares += (double)input[index] * input[index];
        }

        var scale = 1.0f / MathF.Sqrt((float)(sumOfSquares / input.Length) + epsilon);
        for (var index = 0; index < input.Length; index++)
        {
            output[index] = input[index] * scale * weights[index];
        }
    }

    public static void AddInPlace(Span<float> destination, ReadOnlySpan<float> source)
    {
        if (destination.Length != source.Length)
        {
            throw new ArgumentException("Add buffers must have identical lengths.");
        }

        for (var index = 0; index < destination.Length; index++)
        {
            destination[index] += source[index];
        }
    }

    public static void AddBiasInPlace(Span<float> destination, ReadOnlySpan<float> bias) => AddInPlace(destination, bias);

    public static void SwiGluInPlace(Span<float> gate, ReadOnlySpan<float> up)
    {
        if (gate.Length != up.Length)
        {
            throw new ArgumentException("SwiGLU buffers must have identical lengths.");
        }

        for (var index = 0; index < gate.Length; index++)
        {
            var value = gate[index];
            gate[index] = value / (1.0f + MathF.Exp(-value)) * up[index];
        }
    }

    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
        {
            throw new ArgumentException("ArgMax input cannot be empty.", nameof(values));
        }

        var bestIndex = 0;
        var bestValue = values[0];
        for (var index = 1; index < values.Length; index++)
        {
            if (values[index] > bestValue)
            {
                bestValue = values[index];
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    private static int GetMatrixColumns(GgufTensorInfo tensor)
    {
        ValidateQ8Matrix(tensor);
        var columns = checked((int)tensor.Dimensions[0]);
        if (columns % BlockElements != 0)
        {
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' row width {columns} is not divisible by {BlockElements}.");
        }

        return columns;
    }

    private static int GetMatrixRows(GgufTensorInfo tensor)
    {
        ValidateQ8Matrix(tensor);
        return checked((int)tensor.Dimensions[1]);
    }

    private static void ValidateQ8Matrix(GgufTensorInfo tensor)
    {
        if (tensor.Type != Q8Type || tensor.Dimensions.Length != 2)
        {
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' must be a rank-2 Q8_0 matrix.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReadScale(byte* blockPointer)
    {
        var bits = Unsafe.ReadUnaligned<ushort>(blockPointer);
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }
}
