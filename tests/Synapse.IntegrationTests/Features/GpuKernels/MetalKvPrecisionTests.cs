using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;
using static ManagedCode.Synapse.IntegrationTests.Features.GpuKernels.MetalBackendTests;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>The explicit FP16 KV profile: half the KV bytes, FP32 accumulation, declared tolerance.</summary>
[NotInParallel]
public sealed class MetalKvPrecisionTests
{
    private static readonly int[] ExpectedContinuation = [12095, 13, 1084, 374, 279, 7772, 3283, 304];

    [Test]
    public async Task MetalFp16KvMatchesPinnedContinuationAndTracksReference()
    {
        GpuHardware.RequireMetal();
        using var reference = (Qwen2Model)ModelLoader.LoadSourceForValidation(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions { ContextSize = 64, MaximumParallelism = 8, KernelBackend = KernelBackend.Reference });
        using var half = LoadFp16(contextSize: 64);
        int[] prompt = [.. Enumerable.Range(0, 40).Select(index => (index * 7919 % 150_000) + 100)];

        var result = half.Generate(ReferenceBenchmarkFixture.PromptTokens, ExpectedContinuation.Length);
        var expected = reference.EvaluatePromptLogits(prompt);
        var actual = half.EvaluatePromptLogits(prompt);

        await Assert.That(half.RuntimeProfile).IsEqualTo("metal-qwen2-q8_0xf32-kvf16");
        await Assert.That(result.GeneratedTokens).IsEquivalentTo(ExpectedContinuation);
        await Assert.That(ArgMax(actual)).IsEqualTo(ArgMax(expected));
        await Assert.That(LargestError(expected, actual)).IsLessThanOrEqualTo(Range(expected) * 0.01f);
    }

    [Test]
    public async Task MetalFp16KvSplitDecodeTracksPromptRun()
    {
        GpuHardware.RequireMetal();
        int[] prompt = [.. Enumerable.Range(0, 1_000).Select(index => (index * 7919 % 150_000) + 100)];
        int[] continuation = [12095, 13, 1084, 374];
        using var half = LoadFp16(contextSize: 1_100);

        var incremental = half.EvaluateIncrementalLogits(prompt, continuation);
        var full = half.EvaluatePromptLogits([.. prompt, .. continuation]);

        await Assert.That(ArgMax(incremental)).IsEqualTo(ArgMax(full));
        await Assert.That(LargestError(full, incremental)).IsLessThanOrEqualTo(Range(full) * 0.002f);
    }

    [Test]
    public async Task Fp16KvOnCpuFailsExplicitly()
    {
        var exception = await Assert.That(() => ModelLoader.LoadSourceForValidation(
                ReferenceBenchmarkFixture.GetModelPath(),
                new ModelLoadOptions { ContextSize = 64, KernelBackend = KernelBackend.Managed, KvCachePrecision = KvCachePrecision.Fp16 }))
            .Throws<NotSupportedException>();

        await Assert.That(exception!.Message).Contains("FP16");
    }

    private static Qwen2Model LoadFp16(int contextSize) =>
        (Qwen2Model)ModelLoader.LoadSourceForValidation(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions
            {
                ContextSize = contextSize,
                MaximumParallelism = 4,
                KernelBackend = KernelBackend.Metal,
                KvCachePrecision = KvCachePrecision.Fp16,
            });
}
