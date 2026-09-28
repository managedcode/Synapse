namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>Explicit CPU kernel implementation for quantized dense models (ADR-006).</summary>
public enum CpuKernelBackend
{
    /// <summary>Scalar FP32-activation oracle path with sequential prompt evaluation.</summary>
    Reference,

    /// <summary>C# SIMD kernels: Q8_0 weights times Q8_0 activations with fused, batched regions.</summary>
    Managed,

    /// <summary>The managed orchestration with the Rust Q8_0 kernel behind the versioned C ABI.</summary>
    Native,
}

/// <summary>Stable command-line and evidence names for <see cref="CpuKernelBackend"/>.</summary>
public static class CpuKernelBackendNames
{
    /// <summary>Accepted names in documentation order.</summary>
    public const string Usage = "reference|managed|native";

    /// <summary>Returns the stable lower-case name of a backend.</summary>
    public static string ToName(CpuKernelBackend backend) => backend switch
    {
        CpuKernelBackend.Reference => "reference",
        CpuKernelBackend.Managed => "managed",
        CpuKernelBackend.Native => "native",
        _ => throw new ArgumentOutOfRangeException(nameof(backend), backend, "Unknown CPU kernel backend."),
    };

    /// <summary>Parses an exact lower-case backend name.</summary>
    public static bool TryParse(string? value, out CpuKernelBackend backend)
    {
        switch (value)
        {
            case "reference":
                backend = CpuKernelBackend.Reference;
                return true;
            case "managed":
                backend = CpuKernelBackend.Managed;
                return true;
            case "native":
                backend = CpuKernelBackend.Native;
                return true;
            default:
                backend = default;
                return false;
        }
    }
}
