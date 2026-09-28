using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Loads and verifies every Qwen2 dependency in order: the graph verifies before scratch or KV exists.</summary>
internal static class Qwen2ModelComposition
{
    public static DecoderDimensions ReadDimensions(GgufFile file, ModelLoadOptions options)
    {
        var embedding = file.GetRequiredTensor("token_embd.weight");
        var scaling = RopeScalingResolver.Resolve(file.Metadata, "qwen2", options.RopeScaling);
        var trained = file.GetRequiredInt32("qwen2.context_length");
        return new DecoderDimensions(
            file.GetRequiredInt32("qwen2.block_count"),
            file.GetRequiredInt32("qwen2.embedding_length"),
            file.GetRequiredInt32("qwen2.feed_forward_length"),
            file.GetRequiredInt32("qwen2.attention.head_count"),
            file.GetRequiredInt32("qwen2.attention.head_count_kv"),
            checked((int)embedding.Dimensions[1]),
            RequireContext(options.ContextSize, trained, scaling),
            file.GetRequiredSingle("qwen2.rope.freq_base"),
            file.GetRequiredSingle("qwen2.attention.layer_norm_rms_epsilon"),
            scaling,
            options.KvPageActivation);
    }

    /// <summary>The model context limit: trained, or extended by an explicit scaling profile (ADR-013).</summary>
    public static int ModelContextLimit(GgufFile file, DecoderDimensions dimensions) =>
        dimensions.RopeScaling?.ExtendedContextLength ?? file.GetRequiredInt32("qwen2.context_length");

    public static ModelGraph BuildVerifiedGraph(GgufFile file, DecoderDimensions dimensions)
    {
        var graph = Qwen2GraphBuilder.Build(
            file,
            dimensions.LayerCount,
            dimensions.HiddenSize,
            dimensions.FeedForwardSize,
            dimensions.KvWidth,
            ModelContextLimit(file, dimensions),
            dimensions.RopeScaling);
        var verification = ModelGraphVerifier.Verify(graph);
        if (!verification.IsValid)
        {
            throw new InvalidDataException(
                "Qwen2 model graph verification failed: " +
                string.Join(" | ", verification.Diagnostics.Select(diagnostic => diagnostic.Message)));
        }

        return graph;
    }

    private static int RequireContext(int requested, int trained, RopeScaling? scaling)
    {
        if (scaling is not null && scaling.OriginalContextLength != trained)
        {
            throw new NotSupportedException(
                $"The {scaling.Name} profile starts from {scaling.OriginalContextLength} positions, " +
                $"but the model was trained with {trained}.");
        }

        var limit = scaling?.ExtendedContextLength ?? trained;
        return requested <= limit
            ? requested
            : throw new NotSupportedException(scaling is null
                ? $"Requested context {requested} exceeds the model's trained context {trained}. " +
                  "Extend it explicitly with a RoPE scaling profile (for example `--rope-scaling yarn:4`)."
                : $"Requested context {requested} exceeds {limit}, the trained context {trained} extended by {scaling.Name}.");
    }

    /// <summary>Prefaults densely read weights; the token embedding is read by row and stays lazily mapped.</summary>
    public static void PrefetchWeights(GgufFile file, CpuWorkerPool? pool)
    {
        if (pool is null)
        {
            return;
        }

        var work = new MappedPagePrefetchWork(file.GetTensorDataRanges(["token_embd.weight"]), pool.ThreadCount);
        if (work.PageCount > 0)
        {
            _ = pool.Start(work);
        }
    }

    public static IDecoderExecutor CreateExecutor(
        GgufFile file,
        Qwen2Weights weights,
        DecoderDimensions dimensions,
        ModelLoadOptions options,
        CpuWorkerPool? pool)
    {
        if (options.KernelBackend == KernelBackend.Reference)
        {
            return new Qwen2ReferenceExecutor(file, weights, dimensions, options.MaximumParallelism);
        }

        if (KernelBackendNames.IsGpu(options.KernelBackend))
        {
            // The CPU pool only prefaulted the mapped weights for the GPU; it is released once that finishes.
            pool?.WaitForBackground();
            pool?.Dispose();
            var gpuCapacity = DecoderStepCapacity.Create(
                options.GpuPrefillChunkTokens,
                dimensions.ContextSize,
                options.MaximumConcurrentSessions,
                options.ScoringRowsPerStep);
            return new GpuDecoderExecutor(
                file,
                Qwen2DecoderLayout.Describe(file, dimensions),
                options.KernelBackend,
                options.KvCachePrecision,
                gpuCapacity);
        }

        var (kernel, profile) = options.KernelBackend switch
        {
            KernelBackend.Managed => ((Q8MatrixKernel)ManagedQ8Kernel.CreateBest(), "managed-simd-qwen2-q8_0xq8_0"),
            KernelBackend.Native => (NativeQ8Kernel.LoadFromApplicationDirectory(), "native-rust-qwen2-q8_0xq8_0"),
            KernelBackend.Reference or KernelBackend.Metal or KernelBackend.Cuda or _ => throw new ArgumentOutOfRangeException(nameof(options), options.KernelBackend, "Not a CPU backend."),
        };
        var capacity = DecoderStepCapacity.Create(
            options.PrefillChunkTokens,
            dimensions.ContextSize,
            options.MaximumConcurrentSessions,
            options.ScoringRowsPerStep);
        return new Qwen2CpuExecutor(file, weights, dimensions, kernel, profile, pool!, capacity);
    }
}
