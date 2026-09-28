using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>A row-major GGML Q8_0 matrix view over memory-mapped weights: FP16 scale plus 32 codes per block.</summary>
internal readonly unsafe struct Q8Matrix
{
    public const int BlockElements = 32;
    public const int BlockBytes = 34;

    /// <summary>Rows with at least this many blocks stream one row at a time; measured faster than four-row tiles.</summary>
    public const int LongRowBlocks = 64;
    private const uint GgmlQ8ZeroType = 8;

    public Q8Matrix(byte* data, int rows, int columns)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (columns % BlockElements != 0)
        {
            throw new NotSupportedException($"Q8_0 row width {columns} is not divisible by {BlockElements}.");
        }

        Data = data;
        Rows = rows;
        Columns = columns;
        BlocksPerRow = columns / BlockElements;
        RowBytes = BlocksPerRow * BlockBytes;
    }

    public byte* Data { get; }

    public int Rows { get; }

    public int Columns { get; }

    public int BlocksPerRow { get; }

    public int RowBytes { get; }

    public static Q8Matrix FromTensor(GgufFile file, GgufTensorInfo tensor)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.Type != GgmlQ8ZeroType || tensor.Dimensions.Length != 2)
        {
            throw new NotSupportedException($"Tensor '{tensor.Name}' must be a rank-2 Q8_0 matrix.");
        }

        return new Q8Matrix(
            file.GetTensorPointer(tensor),
            checked((int)tensor.Dimensions[1]),
            checked((int)tensor.Dimensions[0]));
    }
}
