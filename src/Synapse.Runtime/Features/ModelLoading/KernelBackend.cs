namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>Explicit kernel implementation for quantized dense models (ADR-006 CPU, ADR-012 GPU).</summary>
public enum KernelBackend
{
    /// <summary>Scalar FP32-activation oracle path with sequential prompt evaluation.</summary>
    Reference,

    /// <summary>C# SIMD kernels: Q8_0 weights times Q8_0 activations with fused, batched regions.</summary>
    Managed,

    /// <summary>The managed orchestration with the Rust Q8_0 kernel behind the versioned C ABI.</summary>
    Native,

    /// <summary>Apple Metal: Q8_0 weights times FP32 activations with flash attention on the GPU.</summary>
    Metal,

    /// <summary>NVIDIA CUDA; fails explicitly until a CUDA build and device are present.</summary>
    Cuda,
}

/// <summary>Stable command-line and evidence names for <see cref="KernelBackend"/>.</summary>
public static class KernelBackendNames
{
    /// <summary>Accepted names in documentation order.</summary>
    public const string Usage = "reference|managed|native|metal|cuda";

    /// <summary>Returns the stable lower-case name of a backend.</summary>
    public static string ToName(KernelBackend backend) => backend switch
    {
        KernelBackend.Reference => "reference",
        KernelBackend.Managed => "managed",
        KernelBackend.Native => "native",
        KernelBackend.Metal => "metal",
        KernelBackend.Cuda => "cuda",
        _ => throw new ArgumentOutOfRangeException(nameof(backend), backend, "Unknown kernel backend."),
    };

    /// <summary>Parses an exact lower-case backend name.</summary>
    public static bool TryParse(string? value, out KernelBackend backend)
    {
        foreach (var candidate in Enum.GetValues<KernelBackend>())
        {
            if (string.Equals(value, ToName(candidate), StringComparison.Ordinal))
            {
                backend = candidate;
                return true;
            }
        }

        backend = default;
        return false;
    }

    /// <summary>Whether the backend executes on a GPU device.</summary>
    public static bool IsGpu(KernelBackend backend) => backend is KernelBackend.Metal or KernelBackend.Cuda;
}
