using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>Explicit execution limits and kernel selection for one loaded model instance.</summary>
public sealed record ModelLoadOptions
{
    /// <summary>Maximum prompt plus generated tokens held by this instance.</summary>
    public int ContextSize { get; init; } = 512;

    /// <summary>Total compute threads, including the calling thread.</summary>
    public int MaximumParallelism { get; init; } = Environment.ProcessorCount;

    /// <summary>CPU kernel implementation; see ADR-006.</summary>
    public CpuKernelBackend KernelBackend { get; init; } = CpuKernelBackend.Managed;

    /// <summary>Sessions that <c>GenerateAsync</c> batches together; further requests wait (ADR-007).</summary>
    public int MaximumConcurrentSessions { get; init; } = 4;

    /// <summary>Prompt tokens evaluated per shared weight pass by optimized backends.</summary>
    internal int PrefillChunkTokens { get; init; } = 64;

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ContextSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumParallelism);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PrefillChunkTokens);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(PrefillChunkTokens, 1024);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumConcurrentSessions);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaximumConcurrentSessions, 64);
        if (!Enum.IsDefined(KernelBackend))
        {
            throw new ArgumentOutOfRangeException(nameof(KernelBackend), KernelBackend, "Unknown CPU kernel backend.");
        }
    }
}
