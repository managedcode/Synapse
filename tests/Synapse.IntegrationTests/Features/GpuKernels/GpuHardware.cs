using System.Runtime.InteropServices;
using ManagedCode.Synapse.Runtime.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>
/// Hardware gates for GPU tests. A machine without the device skips with <c>not_run_missing_hardware</c>; it is
/// never reported as passed. On Apple silicon a probe failure is a real failure, not a skip.
/// </summary>
internal static class GpuHardware
{
    public static GpuDeviceInfo RequireMetal()
    {
        Skip.Unless(
            OperatingSystem.IsMacOS() && RuntimeInformation.ProcessArchitecture == Architecture.Arm64,
            "not_run_missing_hardware: the Metal backend needs an Apple silicon Mac.");
        var device = GpuDevices.Probe(KernelBackend.Metal);
        Skip.When(
            device.AppleFamily < 7,
            $"not_run_missing_hardware: '{device.Name}' reports Apple GPU family {device.AppleFamily}; Apple7+ is required.");
        return device;
    }
}
