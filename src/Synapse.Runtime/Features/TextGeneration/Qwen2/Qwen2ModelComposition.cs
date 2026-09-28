using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Loads and verifies every Qwen2 dependency in order: the graph verifies before scratch or KV exists.</summary>
internal static class Qwen2ModelComposition
{
    public static Qwen2Dimensions ReadDimensions(GgufFile file, ModelLoadOptions options)
    {
        var embedding = file.GetRequiredTensor("token_embd.weight");
        return new Qwen2Dimensions(
            file.GetRequiredInt32("qwen2.block_count"),
            file.GetRequiredInt32("qwen2.embedding_length"),
            file.GetRequiredInt32("qwen2.feed_forward_length"),
            file.GetRequiredInt32("qwen2.attention.head_count"),
            file.GetRequiredInt32("qwen2.attention.head_count_kv"),
            checked((int)embedding.Dimensions[1]),
            Math.Min(options.ContextSize, file.GetRequiredInt32("qwen2.context_length")),
            file.GetRequiredSingle("qwen2.rope.freq_base"),
            file.GetRequiredSingle("qwen2.attention.layer_norm_rms_epsilon"));
    }

    public static ModelGraph BuildVerifiedGraph(GgufFile file, Qwen2Dimensions dimensions)
    {
        var graph = Qwen2GraphBuilder.Build(
            file,
            dimensions.LayerCount,
            dimensions.HiddenSize,
            dimensions.FeedForwardSize,
            dimensions.KvWidth,
            file.GetRequiredInt32("qwen2.context_length"));
        var verification = ModelGraphVerifier.Verify(graph);
        if (!verification.IsValid)
        {
            throw new InvalidDataException(
                "Qwen2 model graph verification failed: " +
                string.Join(" | ", verification.Diagnostics.Select(diagnostic => diagnostic.Message)));
        }

        return graph;
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

    public static IQwen2Executor CreateExecutor(
        GgufFile file,
        Qwen2Weights weights,
        Qwen2Dimensions dimensions,
        ModelLoadOptions options,
        CpuWorkerPool? pool)
    {
        if (options.KernelBackend == CpuKernelBackend.Reference)
        {
            return new Qwen2ReferenceExecutor(file, weights, dimensions, options.MaximumParallelism);
        }

        var (kernel, profile) = options.KernelBackend switch
        {
            CpuKernelBackend.Managed => ((Q8MatrixKernel)ManagedQ8Kernel.CreateBest(), "managed-simd-qwen2-q8_0xq8_0"),
            CpuKernelBackend.Native => (NativeQ8Kernel.LoadFromApplicationDirectory(), "native-rust-qwen2-q8_0xq8_0"),
            CpuKernelBackend.Reference => throw new InvalidOperationException("Reference is handled above."),
            _ => throw new ArgumentOutOfRangeException(nameof(options)),
        };
        var capacity = Qwen2CpuCapacity.Create(
            options.PrefillChunkTokens,
            dimensions.ContextSize,
            options.MaximumConcurrentSessions);
        return new Qwen2CpuExecutor(file, weights, dimensions, kernel, profile, pool!, capacity);
    }
}
