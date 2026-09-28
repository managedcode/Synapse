using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// Persistent compute threads for one model. <see cref="ThreadCount"/> includes the calling thread, which runs
/// worker 0 of every region. Idle workers spin briefly between regions, then block until the next dispatch.
/// </summary>
internal sealed class CpuWorkerPool : IDisposable
{
    private static readonly long SpinBudgetTicks = Stopwatch.Frequency / 1_000;
    private readonly Thread[] _threads;
    private readonly object _gate = new();
    private CpuParallelWork? _work;
    private ExceptionDispatchInfo? _failure;
    private int _generation;
    private int _pending;
    private int _sleeping;
    private volatile bool _disposed;
    private bool _backgroundActive;

    public CpuWorkerPool(int threadCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threadCount);
        ThreadCount = threadCount;
        _threads = new Thread[threadCount - 1];
        for (var index = 0; index < _threads.Length; index++)
        {
            _threads[index] = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = $"synapse-cpu-{index + 1}",
            };
            _threads[index].Start(index + 1);
        }
    }

    public int ThreadCount { get; }

    /// <summary>Runs <paramref name="work"/> on every thread and returns after all of them finish.</summary>
    public void Run(CpuParallelWork work)
    {
        ArgumentNullException.ThrowIfNull(work);
        ObjectDisposedException.ThrowIf(_disposed, this);
        WaitForBackground();
        if (_threads.Length == 0)
        {
            work.Execute(0);
            return;
        }

        Dispatch(work);
        try
        {
            work.Execute(0);
        }
        catch (Exception exception)
        {
            RecordFailure(exception);
        }

        Complete();
    }

    /// <summary>
    /// Starts <paramref name="work"/> on the worker threads only and returns immediately, so the caller can do
    /// sequential work meanwhile. The next <see cref="Run"/> or <see cref="WaitForBackground"/> joins it.
    /// Returns <see langword="false"/> when the pool has no worker threads.
    /// </summary>
    public bool Start(CpuParallelWork work)
    {
        ArgumentNullException.ThrowIfNull(work);
        ObjectDisposedException.ThrowIf(_disposed, this);
        WaitForBackground();
        if (_threads.Length == 0)
        {
            return false;
        }

        Dispatch(work);
        _backgroundActive = true;
        return true;
    }

    /// <summary>Waits for work begun by <see cref="Start"/> and rethrows its first failure.</summary>
    public void WaitForBackground()
    {
        if (!_backgroundActive)
        {
            return;
        }

        _backgroundActive = false;
        Complete();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (_backgroundActive)
        {
            _backgroundActive = false;
            WaitForWorkers();
            _work = null;
        }

        _disposed = true;
        lock (_gate)
        {
            Monitor.PulseAll(_gate);
        }

        foreach (var thread in _threads)
        {
            thread.Join();
        }
    }

    private void WorkerLoop(object? state)
    {
        var worker = (int)state!;
        var seen = 0;
        while (true)
        {
            seen = WaitForGeneration(seen);
            if (_disposed)
            {
                return;
            }

            try
            {
                _work!.Execute(worker);
            }
            catch (Exception exception)
            {
                RecordFailure(exception);
            }

            _ = Interlocked.Decrement(ref _pending);
        }
    }

    private int WaitForGeneration(int seen)
    {
        var start = Stopwatch.GetTimestamp();
        for (var spins = 1; ; spins++)
        {
            var generation = Volatile.Read(ref _generation);
            if (generation != seen || _disposed)
            {
                return generation;
            }

            if ((spins & 63) == 0 && Stopwatch.GetTimestamp() - start > SpinBudgetTicks)
            {
                return SleepUntilGeneration(seen);
            }

            Thread.SpinWait(8);
        }
    }

    private int SleepUntilGeneration(int seen)
    {
        lock (_gate)
        {
            _ = Interlocked.Increment(ref _sleeping);
            try
            {
                int generation;
                while ((generation = Volatile.Read(ref _generation)) == seen && !_disposed)
                {
                    _ = Monitor.Wait(_gate);
                }

                return generation;
            }
            finally
            {
                _ = Interlocked.Decrement(ref _sleeping);
            }
        }
    }

    private void Dispatch(CpuParallelWork work)
    {
        _work = work;
        Volatile.Write(ref _pending, _threads.Length);
        _ = Interlocked.Increment(ref _generation);
        if (Volatile.Read(ref _sleeping) > 0)
        {
            lock (_gate)
            {
                Monitor.PulseAll(_gate);
            }
        }
    }

    private void Complete()
    {
        WaitForWorkers();
        _work = null;
        var failure = Interlocked.Exchange(ref _failure, null);
        failure?.Throw();
    }

    private void WaitForWorkers()
    {
        var spinner = default(SpinWait);
        while (Volatile.Read(ref _pending) != 0)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }
    }

    private void RecordFailure(Exception exception) =>
        _ = Interlocked.CompareExchange(ref _failure, ExceptionDispatchInfo.Capture(exception), null);
}
