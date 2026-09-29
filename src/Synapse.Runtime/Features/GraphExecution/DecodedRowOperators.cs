using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution;

/// <summary>
/// Matrix rows the reference path reads in encodings other than Q8_0: F32 and GGML Q4_K and Q6_K (ADR-021). Each
/// row is decoded exactly as ggml does, then dotted with the input in index order, so an F32 model that holds the
/// decoded values gives bitwise the same results.
/// </summary>
internal static unsafe class DecodedRowOperators
{
    public static void ReadRow(GgufFile file, GgufTensorInfo tensor, int row, Span<float> destination)
    {
        var (decoder, rows, rowBytes) = Describe(tensor, destination.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)row, (uint)rows, nameof(row));
        decoder.Decode(new ReadOnlySpan<byte>(file.GetTensorPointer(tensor) + (row * rowBytes), rowBytes), destination);
    }

    public static void Multiply(GgufFile file, GgufTensorInfo tensor, float[] input, float[] output, ParallelOptions options)
    {
        var (decoder, rows, rowBytes) = Describe(tensor, input.Length);
        if (output.Length != rows)
        {
            throw new ArgumentException($"Tensor '{tensor.Name}' has {rows} rows, not {output.Length}.");
        }

        var address = (nint)file.GetTensorPointer(tensor);
        _ = Parallel.For(0, rows, options, () => new float[input.Length], (row, _, values) =>
        {
            decoder.Decode(new ReadOnlySpan<byte>((byte*)address + ((long)row * rowBytes), rowBytes), values);
            var sum = 0.0f;
            for (var index = 0; index < values.Length; index++)
            {
                sum += values[index] * input[index];
            }

            output[row] = sum;
            return values;
        }, _ => { });
    }

    private static (ISourceTensorDecoder Decoder, int Rows, int RowBytes) Describe(GgufTensorInfo tensor, int columns)
    {
        if (tensor.Dimensions.Length != 2 || (long)tensor.Dimensions[0] != columns)
        {
            throw new ArgumentException($"Tensor '{tensor.Name}' is not a rank-2 matrix of {columns} columns.");
        }

        var decoder = SourceEncodings.GetGguf(tensor.Type);
        return (decoder, checked((int)tensor.Dimensions[1]), checked((int)decoder.GetByteLength(columns)));
    }
}
