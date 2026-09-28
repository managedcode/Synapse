using System.Runtime.CompilerServices;
namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>One parallel region executed by every thread of a <see cref="CpuWorkerPool"/>.</summary>
internal abstract class CpuParallelWork
{
    /// <summary>Runs this thread's share; <paramref name="worker"/> 0 is the calling thread.</summary>
    internal abstract void Execute(int worker);
}

/// <summary>
/// Row-parallel work split into dynamic chunks that threads claim with one atomic counter, so slower cores
/// take fewer chunks. Chunk starts stay aligned for four-row kernels.
/// </summary>
internal abstract class ChunkedRowWork : CpuParallelWork
{
    public const int MaximumChunkRows = 4096;
    private const int ChunksPerThread = 2;
    private int _rows;
    private int _chunkRows;
    private int _chunkCount;
    private int _nextChunk;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal sealed override void Execute(int worker)
    {
        while (true)
        {
            var chunk = Interlocked.Increment(ref _nextChunk) - 1;
            if (chunk >= _chunkCount)
            {
                return;
            }

            var start = chunk * _chunkRows;
            ProcessRows(worker, start, Math.Min(_chunkRows, _rows - start));
        }
    }

    /// <summary>Prepares a new region. Call before each <see cref="CpuWorkerPool.Run"/>.</summary>
    protected void Reset(
        int rows,
        int threadCount,
        int alignment = 4,
        int minimumChunkRows = 16,
        int maximumChunkRows = MaximumChunkRows)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threadCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(alignment);
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumChunkRows, alignment);
        var target = threadCount * ChunksPerThread;
        var chunkRows = Math.Clamp(
            (rows + target - 1) / target,
            Math.Min(minimumChunkRows, maximumChunkRows),
            maximumChunkRows);
        chunkRows = Math.Min(
            (chunkRows + alignment - 1) / alignment * alignment,
            maximumChunkRows / alignment * alignment);
        _rows = rows;
        _chunkRows = chunkRows;
        _chunkCount = (rows + chunkRows - 1) / chunkRows;
        Volatile.Write(ref _nextChunk, 0);
    }

    protected abstract void ProcessRows(int worker, int rowStart, int rowCount);
}
