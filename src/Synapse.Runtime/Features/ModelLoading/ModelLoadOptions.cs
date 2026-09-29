using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>Explicit execution limits and kernel selection for one loaded model instance.</summary>
public sealed record ModelLoadOptions
{
    /// <summary>Maximum prompt plus generated tokens held by this instance.</summary>
    public int ContextSize { get; init; } = 512;

    /// <summary>Total compute threads, including the calling thread.</summary>
    public int MaximumParallelism { get; init; } = Environment.ProcessorCount;

    /// <summary>Kernel implementation; see ADR-006 (CPU) and ADR-012 (GPU).</summary>
    public KernelBackend KernelBackend { get; init; } = KernelBackend.Managed;

    /// <summary>Sessions that <c>GenerateAsync</c> batches together; further requests wait (ADR-007).</summary>
    public int MaximumConcurrentSessions { get; init; } = 4;

    /// <summary>
    /// Explicit rotary scaling that extends the model beyond its trained context, or <see langword="null"/>.
    /// A context above the trained window without a covering profile fails at load (ADR-013).
    /// </summary>
    public RopeScaling? RopeScaling { get; init; }

    /// <summary>KV cache element type. FP16 is a GPU-only profile; CPU backends fail explicitly (ADR-012).</summary>
    public KvCachePrecision KvCachePrecision { get; init; } = KvCachePrecision.Fp32;

    /// <summary>
    /// Full-vocabulary logit rows one step may return while scoring (ADR-015). Each row costs vocabulary × 4 bytes in
    /// the host buffer and again on a GPU device.
    /// </summary>
    public int ScoringRowsPerStep { get; init; } = 8;

    /// <summary>KV slots grow in multiples of this many positions, at least doubling (ADR-017).</summary>
    internal int KvGrowthPositions { get; init; } = 1024;

    /// <summary>
    /// Query-aware KV page activation for decode tokens (ADR-016), or <see langword="null"/> for dense attention. An
    /// approximation profile implemented on the managed and native CPU backends; other backends fail at load.
    /// </summary>
    public KvPageActivation? KvPageActivation { get; init; }

    /// <summary>
    /// Whole layers that do not run (ADR-019), or <see langword="null"/> for the dense model. An approximation
    /// profile: every backend runs the kept layers only, and dropped weights and KV are never touched.
    /// </summary>
    public LayerDropProfile? LayerDrop { get; init; }

    /// <summary>
    /// Reuses the direct session's K and V for the longest token prefix shared with the previous direct request
    /// (ADR-018). Reused positions carry the earlier request's numbers, so this is opt-in.
    /// </summary>
    public bool ReusePromptPrefix { get; init; }

    /// <summary>Prompt tokens evaluated per shared weight pass by optimized backends.</summary>
    internal int PrefillChunkTokens { get; init; } = 64;

    /// <summary>Prompt tokens per GPU step; larger chunks amortize dispatch and synchronization (ADR-012).</summary>
    internal int GpuPrefillChunkTokens { get; init; } = 512;

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ContextSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumParallelism);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(PrefillChunkTokens);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(PrefillChunkTokens, 1024);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(GpuPrefillChunkTokens);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(GpuPrefillChunkTokens, 4096);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumConcurrentSessions);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaximumConcurrentSessions, 64);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ScoringRowsPerStep);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(ScoringRowsPerStep, 512);
        if (KvGrowthPositions <= 0 || KvGrowthPositions % 64 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(KvGrowthPositions), KvGrowthPositions, "KV growth is a positive multiple of 64 positions.");
        }

        if (!Enum.IsDefined(KernelBackend))
        {
            throw new ArgumentOutOfRangeException(nameof(KernelBackend), KernelBackend, "Unknown kernel backend.");
        }

        if (!Enum.IsDefined(KvCachePrecision))
        {
            throw new ArgumentOutOfRangeException(nameof(KvCachePrecision), KvCachePrecision, "Unknown KV precision.");
        }

        KvPageActivation?.Validate();
        if (KvPageActivation is not null && KernelBackend is not (KernelBackend.Managed or KernelBackend.Native))
        {
            throw new NotSupportedException(
                $"KV page activation (ADR-016) runs on the managed and native CPU backends; the {KernelBackendNames.ToName(KernelBackend)} backend does not implement it yet.");
        }

        if (KvCachePrecision != KvCachePrecision.Fp32 && !KernelBackendNames.IsGpu(KernelBackend))
        {
            throw new NotSupportedException(
                $"The FP16 KV cache is a GPU profile; the {KernelBackendNames.ToName(KernelBackend)} backend keeps FP32 KV.");
        }
    }
}
