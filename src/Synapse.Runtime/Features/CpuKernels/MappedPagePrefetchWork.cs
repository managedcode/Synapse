namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// Touches one byte per page of read-only mapped ranges so first-use page faults happen on idle workers while
/// the caller verifies the model graph, instead of inside the first forward pass. Callers pass only densely
/// read tensors; sparse tables such as token embeddings stay unmapped to keep resident memory low.
/// </summary>
internal sealed unsafe class MappedPagePrefetchWork : ChunkedRowWork
{
    private readonly (nint Start, long Length)[] _ranges;
    private readonly long[] _firstPages;
    private readonly int _pageSize;
    private long _checksum;

    public MappedPagePrefetchWork(IReadOnlyList<(nint Start, long Length)> ranges, int threadCount)
    {
        ArgumentNullException.ThrowIfNull(ranges);
        _pageSize = Environment.SystemPageSize;
        _ranges = [.. ranges.Where(range => range.Start != 0 && range.Length > 0)];
        _firstPages = new long[_ranges.Length];
        long pages = 0;
        for (var index = 0; index < _ranges.Length; index++)
        {
            _firstPages[index] = pages;
            pages += (_ranges[index].Length + _pageSize - 1) / _pageSize;
        }

        PageCount = checked((int)pages);
        if (PageCount > 0)
        {
            Reset(PageCount, Math.Max(1, threadCount - 1), alignment: 1, minimumChunkRows: 64);
        }
    }

    public int PageCount { get; }

    /// <summary>Sum of touched bytes; read after completion so the loads cannot be elided.</summary>
    public long Checksum => Interlocked.Read(ref _checksum);

    protected override void ProcessRows(int worker, int rowStart, int rowCount)
    {
        long sum = 0;
        var range = Array.BinarySearch(_firstPages, rowStart);
        range = range >= 0 ? range : ~range - 1;
        for (var page = (long)rowStart; page < rowStart + rowCount; page++)
        {
            while (range + 1 < _firstPages.Length && page >= _firstPages[range + 1])
            {
                range++;
            }

            var (start, length) = _ranges[range];
            var offset = Math.Min((page - _firstPages[range]) * _pageSize, length - 1);
            sum += Volatile.Read(ref ((byte*)start)[offset]);
        }

        _ = Interlocked.Add(ref _checksum, sum);
    }
}
