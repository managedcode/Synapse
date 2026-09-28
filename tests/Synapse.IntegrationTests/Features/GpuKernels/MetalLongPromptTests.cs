using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;
using static ManagedCode.Synapse.IntegrationTests.Features.GpuKernels.MetalBackendTests;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>Prompt-run GEMM, multi-chunk prefill, and split decode attention on Metal.</summary>
[NotInParallel]
public sealed class MetalLongPromptTests
{
    [Test]
    public async Task MetalPromptRunsTrackReference()
    {
        GpuHardware.RequireMetal();
        int[] prompt = [.. Enumerable.Range(0, 40).Select(index => (index * 7919 % 150_000) + 100)];
        using var reference = (Qwen2Model)ModelLoader.Load(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions { ContextSize = 64, MaximumParallelism = 8, KernelBackend = KernelBackend.Reference });
        using var metal = Load(contextSize: 64);

        var expected = reference.EvaluatePromptLogits(prompt);
        var actual = metal.EvaluatePromptLogits(prompt);

        await Assert.That(ArgMax(actual)).IsEqualTo(ArgMax(expected));
        await Assert.That(LargestError(expected, actual)).IsLessThanOrEqualTo(Range(expected) * 0.002f);
    }

    [Test]
    public async Task MetalLongPromptAcrossChunksTracksManagedCpu()
    {
        GpuHardware.RequireMetal();
        int[] prompt = [.. Enumerable.Range(0, 1_100).Select(index => (index * 7919 % 150_000) + 100)];
        using var managed = (Qwen2Model)ModelLoader.Load(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions { ContextSize = 1_200, MaximumParallelism = 8, KernelBackend = KernelBackend.Managed });
        using var metal = Load(contextSize: 1_200);

        var expected = managed.EvaluatePromptLogits(prompt);
        var actual = metal.EvaluatePromptLogits(prompt);

        await Assert.That(ArgMax(actual)).IsEqualTo(ArgMax(expected));
        await Assert.That(LargestError(expected, actual)).IsLessThanOrEqualTo(Range(expected) * 0.05f);
    }

    [Test]
    public async Task MetalSplitDecodeTracksPromptRunAttention()
    {
        GpuHardware.RequireMetal();
        int[] prompt = [.. Enumerable.Range(0, 1_000).Select(index => (index * 7919 % 150_000) + 100)];
        int[] continuation = [12095, 13, 1084, 374];
        using var metal = Load(contextSize: 1_100);

        var incremental = metal.EvaluateIncrementalLogits(prompt, continuation);
        var full = metal.EvaluatePromptLogits([.. prompt, .. continuation]);

        await Assert.That(ArgMax(incremental)).IsEqualTo(ArgMax(full));
        await Assert.That(LargestError(full, incremental)).IsLessThanOrEqualTo(Range(full) * 0.002f);
    }
}
