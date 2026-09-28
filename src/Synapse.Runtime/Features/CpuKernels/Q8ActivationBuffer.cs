using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>Pinned Q8_0 activation rows for up to <see cref="Capacity"/> tokens.</summary>
internal sealed unsafe class Q8ActivationBuffer
{
    private readonly sbyte[] _quants;
    private readonly float[] _scales;

    public Q8ActivationBuffer(int columns, int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        if (columns % Q8ActivationQuantizer.BlockElements != 0)
        {
            throw new ArgumentException(
                $"Q8_0 activation width {columns} is not divisible by {Q8ActivationQuantizer.BlockElements}.",
                nameof(columns));
        }

        Columns = columns;
        BlocksPerRow = columns / Q8ActivationQuantizer.BlockElements;
        Capacity = capacity;
        _quants = GC.AllocateArray<sbyte>(checked(columns * capacity), pinned: true);
        _scales = GC.AllocateArray<float>(checked(BlocksPerRow * capacity), pinned: true);
    }

    public int Columns { get; }

    public int BlocksPerRow { get; }

    public int Capacity { get; }

    /// <summary>Stable address of token 0's codes; the array lives on the pinned object heap.</summary>
    public sbyte* Quants => (sbyte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_quants));

    /// <summary>Stable address of token 0's block scales.</summary>
    public float* Scales => (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_scales));

    public void Quantize(int token, ReadOnlySpan<float> row)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)token, (uint)Capacity, nameof(token));
        Q8ActivationQuantizer.QuantizeRow(
            row,
            _quants.AsSpan(token * Columns, Columns),
            _scales.AsSpan(token * BlocksPerRow, BlocksPerRow));
    }

    public ReadOnlySpan<sbyte> GetQuants(int token) => _quants.AsSpan(checked(token * Columns), Columns);

    public ReadOnlySpan<float> GetScales(int token) => _scales.AsSpan(checked(token * BlocksPerRow), BlocksPerRow);
}
