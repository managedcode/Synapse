using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Runtime.Features.GpuKernels;

/// <summary>A GPU device as reported by its backend (ADR-012).</summary>
/// <param name="Backend">Stable backend name, for example <c>metal</c>.</param>
/// <param name="Name">Device name reported by the driver.</param>
/// <param name="RecommendedWorkingSetBytes">Memory the device can use without degrading performance.</param>
/// <param name="MaximumBufferBytes">Largest single device buffer.</param>
/// <param name="UnifiedMemory">Whether CPU and GPU share physical memory.</param>
/// <param name="AppleFamily">Highest Apple GPU family (7 = M1), or 0 for other vendors.</param>
public sealed record GpuDeviceInfo(
    string Backend,
    string Name,
    long RecommendedWorkingSetBytes,
    long MaximumBufferBytes,
    bool UnifiedMemory,
    int AppleFamily);

/// <summary>Discovers GPU devices through the native GPU library.</summary>
public static class GpuDevices
{
    /// <summary>
    /// Describes the default device of a GPU backend. Throws <see cref="NotSupportedException"/> when the library,
    /// backend, or device is unavailable; it never reports a missing device as present.
    /// </summary>
    public static GpuDeviceInfo Probe(KernelBackend backend)
    {
        if (!KernelBackendNames.IsGpu(backend))
        {
            throw new ArgumentOutOfRangeException(nameof(backend), backend, "Not a GPU backend.");
        }

        return NativeGpuLibrary.LoadFromApplicationDirectory().Probe(backend);
    }
}
