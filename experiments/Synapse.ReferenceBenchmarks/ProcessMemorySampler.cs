using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

internal sealed class ProcessMemorySampler : IAsyncDisposable
{
    private readonly Process _process;
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _sampling;
    private TimeSpan _cpu;
    private long? _residentPeak;
    private long? _privateVirtualPeak;
    private long? _virtualPeak;
    private long? _physicalFootprintPeak;
    private int _samples;

    public ProcessMemorySampler(Process process)
    {
        _process = process;
        Sample();
        _sampling = SampleUntilStoppedAsync();
    }

    public async Task<ObservedProcessMetrics> CompleteAsync()
    {
        await _stop.CancelAsync().ConfigureAwait(false);
        await _sampling.ConfigureAwait(false);
        Sample();
        return new ObservedProcessMetrics(
            _cpu, _residentPeak, _privateVirtualPeak, _virtualPeak,
            _physicalFootprintPeak, _samples);
    }

    public async ValueTask DisposeAsync()
    {
        await _stop.CancelAsync().ConfigureAwait(false);
        await _sampling.ConfigureAwait(false);
        _stop.Dispose();
    }

    private async Task SampleUntilStoppedAsync()
    {
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(5));
        try
        {
            while (await timer.WaitForNextTickAsync(_stop.Token).ConfigureAwait(false))
            {
                if (_process.HasExited)
                {
                    break;
                }

                Sample();
            }
        }
        catch (OperationCanceledException) when (_stop.IsCancellationRequested)
        {
        }
    }

    private void Sample()
    {
        try
        {
            _process.Refresh();
            _cpu = _process.TotalProcessorTime;
            _residentPeak = MaxPositive(_residentPeak,
                Math.Max(_process.WorkingSet64, _process.PeakWorkingSet64));
            _privateVirtualPeak = MaxPositive(_privateVirtualPeak, _process.PrivateMemorySize64);
            _virtualPeak = MaxPositive(_virtualPeak, _process.VirtualMemorySize64);
            _physicalFootprintPeak = MaxPositive(_physicalFootprintPeak,
                MacProcessFootprint.TryReadBytes(_process.Id));
            _samples++;
        }
        catch (InvalidOperationException)
        {
        }
        catch (Win32Exception)
        {
        }
    }

    private static long? MaxPositive(long? previous, long? current) => current is > 0
        ? previous is > 0 ? Math.Max(previous.Value, current.Value) : current
        : previous;
}

internal sealed record ObservedProcessMetrics(
    TimeSpan Cpu,
    long? MaximumObservedWorkingSetBytes,
    long? MaximumObservedPrivateVirtualBytes,
    long? MaximumObservedVirtualBytes,
    long? PeakPhysicalFootprintBytes,
    int MemorySampleCount);

internal static class MacProcessFootprint
{
    private const int RusageInfoV0 = 0;

    [StructLayout(LayoutKind.Explicit, Size = 96)]
    private struct Rusage
    {
        [FieldOffset(72)]
        public ulong PhysicalFootprint;
    }

#pragma warning disable SYSLIB1054 // The isolated benchmark P/Invoke keeps unsafe code disabled.
    [DllImport("libproc", EntryPoint = "proc_pid_rusage")]
    private static extern int Read(int pid, int flavor, out Rusage usage);
#pragma warning restore SYSLIB1054

    public static long? TryReadBytes(int pid)
    {
        if (!OperatingSystem.IsMacOS() || Read(pid, RusageInfoV0, out var usage) != 0 ||
            usage.PhysicalFootprint is 0 or > long.MaxValue)
        {
            return null;
        }

        return (long)usage.PhysicalFootprint;
    }
}
