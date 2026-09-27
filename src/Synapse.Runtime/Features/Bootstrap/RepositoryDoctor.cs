using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using ManagedCode.Synapse.Contracts.Features.Bootstrap;

namespace ManagedCode.Synapse.Runtime.Features.Bootstrap;

/// <summary>Inspects the local CPU and validates an offline execution budget.</summary>
public static class RepositoryDoctor
{
    /// <summary>Returns observed capabilities without performing network I/O.</summary>
    /// <param name="request">Requested bounded runtime profile.</param>
    /// <returns>A supported profile or a typed pre-execution failure.</returns>
    public static DoctorResult Inspect(DoctorRequest request)
    {
        var runtimeIdentifier = RuntimeInformation.RuntimeIdentifier;
        var architecture = RuntimeInformation.OSArchitecture.ToString();
        var processArchitecture = RuntimeInformation.ProcessArchitecture.ToString();
        var capabilities = new CpuCapabilities(
            Vector.IsHardwareAccelerated,
            AdvSimd.IsSupported,
            Avx2.IsSupported,
            Avx512F.IsSupported);

        if (request.MemoryBudgetBytes <= 0)
        {
            return CreateFailure(
                request,
                architecture,
                processArchitecture,
                runtimeIdentifier,
                capabilities,
                DoctorErrorCode.InvalidMemoryBudget,
                "Memory budget must be a positive signed 64-bit byte count.");
        }

        return RuntimeInformation.OSArchitecture is not Architecture.Arm64 and not Architecture.X64
            ? CreateFailure(
                request,
                architecture,
                processArchitecture,
                runtimeIdentifier,
                capabilities,
                DoctorErrorCode.UnsupportedArchitecture,
                $"Architecture '{architecture}' is not supported by the bootstrap CPU profile.")
            : new DoctorResult(
                true,
                RuntimeInformation.OSDescription,
                architecture,
                processArchitecture,
                runtimeIdentifier,
                request.MemoryBudgetBytes,
                GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
                capabilities,
                null);
    }

    private static DoctorResult CreateFailure(
        DoctorRequest request,
        string architecture,
        string processArchitecture,
        string runtimeIdentifier,
        CpuCapabilities capabilities,
        DoctorErrorCode code,
        string message) => new(
            false,
            RuntimeInformation.OSDescription,
            architecture,
            processArchitecture,
            runtimeIdentifier,
            request.MemoryBudgetBytes,
            GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
            capabilities,
            new DoctorError(code, message));
}
