using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

public sealed class CpuWorkerPoolTests
{
    [Test]
    public async Task WorkerPoolRunsEveryChunkOnce()
    {
        foreach (var threads in new[] { 1, 2, 3, 8 })
        {
            using var pool = new CpuWorkerPool(threads);
            foreach (var rows in new[] { 1, 4, 17, 896, 4864, 151936 })
            {
                var work = new CountingWork(rows);
                for (var run = 0; run < 25; run++)
                {
                    work.Prepare(pool.ThreadCount);
                    pool.Run(work);
                }

                await Assert.That(work.Counts.All(count => count == 25)).IsTrue()
                    .Because($"threads={threads}, rows={rows}");
            }
        }
    }

    [Test]
    public async Task WorkerPoolUsesMultipleThreads()
    {
        using var pool = new CpuWorkerPool(3);
        var work = new CountingWork(256, spinPerRow: 2_000);
        work.Prepare(pool.ThreadCount);

        pool.Run(work);

        await Assert.That(work.ThreadIds.Count).IsGreaterThanOrEqualTo(2);
        await Assert.That(work.ThreadIds.Count).IsLessThanOrEqualTo(3);
    }

    [Test]
    public async Task WorkerPoolRethrowsWorkerFailure()
    {
        using var pool = new CpuWorkerPool(4);
        var failing = new CountingWork(1024, failAtRow: 700);
        failing.Prepare(pool.ThreadCount);

        await Assert.That(() => pool.Run(failing)).Throws<InvalidOperationException>()
            .WithMessageContaining("row 700");

        var healthy = new CountingWork(1024);
        healthy.Prepare(pool.ThreadCount);
        pool.Run(healthy);
        await Assert.That(healthy.Counts.All(count => count == 1)).IsTrue();
    }

    [Test]
    public async Task WorkerPoolRejectsInvalidThreadCountAndUseAfterDispose()
    {
        await Assert.That(() => new CpuWorkerPool(0)).Throws<ArgumentOutOfRangeException>();
        var pool = new CpuWorkerPool(2);
        pool.Dispose();
        var work = new CountingWork(8);
        work.Prepare(2);
        await Assert.That(() => pool.Run(work)).Throws<ObjectDisposedException>();
    }

    private sealed class CountingWork(int rows, int spinPerRow = 0, int failAtRow = -1) : ChunkedRowWork
    {
        public int[] Counts { get; } = new int[rows];

        public System.Collections.Concurrent.ConcurrentDictionary<int, bool> ThreadIds { get; } = new();

        public void Prepare(int threadCount) => Reset(rows, threadCount);

        protected override void ProcessRows(int worker, int rowStart, int rowCount)
        {
            ThreadIds[Environment.CurrentManagedThreadId] = true;
            for (var row = rowStart; row < rowStart + rowCount; row++)
            {
                if (row == failAtRow)
                {
                    throw new InvalidOperationException($"Deliberate failure at row {row}.");
                }

                Thread.SpinWait(spinPerRow);
                _ = Interlocked.Increment(ref Counts[row]);
            }
        }
    }
}
