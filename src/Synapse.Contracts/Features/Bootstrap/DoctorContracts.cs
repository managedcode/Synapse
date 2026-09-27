namespace ManagedCode.Synapse.Contracts.Features.Bootstrap;

/// <summary>Inputs for an offline repository and runtime capability inspection.</summary>
/// <param name="MemoryBudgetBytes">Maximum managed Synapse working-set budget in bytes.</param>
public sealed record DoctorRequest(long MemoryBudgetBytes);

/// <summary>Stable bootstrap failure categories returned before runtime execution.</summary>
public enum DoctorErrorCode
{
    /// <summary>The supplied memory budget was zero or negative.</summary>
    InvalidMemoryBudget,

    /// <summary>The current CPU architecture has no supported bootstrap profile.</summary>
    UnsupportedArchitecture,
}

/// <summary>A typed doctor failure with an actionable diagnostic.</summary>
/// <param name="Code">Stable machine-readable failure category.</param>
/// <param name="Message">Human-readable diagnostic.</param>
public sealed record DoctorError(DoctorErrorCode Code, string Message);

/// <summary>CPU vector capabilities relevant to the reference and optimized paths.</summary>
/// <param name="VectorAccelerated">Whether managed vectors use hardware acceleration.</param>
/// <param name="ArmAdvSimd">Whether ARM Advanced SIMD is available.</param>
/// <param name="X86Avx2">Whether x86 AVX2 is available.</param>
/// <param name="X86Avx512F">Whether x86 AVX-512 Foundation is available.</param>
public sealed record CpuCapabilities(
    bool VectorAccelerated,
    bool ArmAdvSimd,
    bool X86Avx2,
    bool X86Avx512F);

/// <summary>Observed offline platform capabilities and validation outcome.</summary>
/// <param name="IsSupported">Whether the local bootstrap CPU profile is supported.</param>
/// <param name="OperatingSystem">Runtime-provided operating-system description.</param>
/// <param name="Architecture">Operating-system architecture.</param>
/// <param name="ProcessArchitecture">Current process architecture.</param>
/// <param name="RuntimeIdentifier">Current .NET runtime identifier.</param>
/// <param name="MemoryBudgetBytes">Validated requested working-set budget.</param>
/// <param name="AvailableMemoryBytes">Memory available to the current .NET process.</param>
/// <param name="Cpu">Observed vector instruction capabilities.</param>
/// <param name="Error">Typed failure when the profile is unsupported.</param>
public sealed record DoctorResult(
    bool IsSupported,
    string OperatingSystem,
    string Architecture,
    string ProcessArchitecture,
    string RuntimeIdentifier,
    long MemoryBudgetBytes,
    long AvailableMemoryBytes,
    CpuCapabilities Cpu,
    DoctorError? Error);
