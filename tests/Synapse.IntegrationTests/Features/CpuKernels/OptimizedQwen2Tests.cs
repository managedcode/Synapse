using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

[NotInParallel]
public sealed class OptimizedQwen2Tests
{
    private static readonly int[] ExpectedContinuation = [12095, 13, 1084, 374, 279, 7772, 3283, 304];

    [Test]
    [Arguments(CpuKernelBackend.Managed, "managed-simd-qwen2-q8_0xq8_0")]
    [Arguments(CpuKernelBackend.Native, "native-rust-qwen2-q8_0xq8_0")]
    public async Task OptimizedBackendMatchesPinnedContinuation(CpuKernelBackend backend, string profile)
    {
        foreach (var threads in new[] { 1, 3 })
        {
            using var model = Load(backend, threads);

            var result = model.Generate(ReferenceBenchmarkFixture.PromptTokens, ExpectedContinuation.Length);

            await Assert.That(model.RuntimeProfile).IsEqualTo(profile);
            await Assert.That(result.GeneratedTokens).IsEquivalentTo(ExpectedContinuation)
                .Because($"threads={threads}");
        }
    }

    [Test]
    public async Task ReferenceBackendRemainsAvailable()
    {
        using var model = Load(CpuKernelBackend.Reference, 2);

        var result = model.Generate(ReferenceBenchmarkFixture.PromptTokens, 2);

        await Assert.That(model.RuntimeProfile).IsEqualTo("reference-qwen2-q8_0");
        await Assert.That(result.GeneratedTokens).IsEquivalentTo(ExpectedContinuation[..2]);
    }

    [Test]
    public async Task BatchedPrefillMatchesSequentialPrefill()
    {
        using var sequential = Load(CpuKernelBackend.Managed, 2, prefillChunkTokens: 1);
        using var paired = Load(CpuKernelBackend.Managed, 2, prefillChunkTokens: 2);
        using var batched = Load(CpuKernelBackend.Managed, 2);
        int[] prompt = [.. ReferenceBenchmarkFixture.PromptTokens, 12095, 13, 1084];

        var expected = sequential.EvaluatePromptLogits(prompt);
        var pairedLogits = paired.EvaluatePromptLogits(prompt);
        var batchedLogits = batched.EvaluatePromptLogits(prompt);

        await Assert.That(Bits(pairedLogits)).IsEquivalentTo(Bits(expected));
        await Assert.That(Bits(batchedLogits)).IsEquivalentTo(Bits(expected));
    }

    [Test]
    public async Task ManagedLogitsTrackReferenceLogits()
    {
        using var reference = Load(CpuKernelBackend.Reference, 4);
        using var managed = Load(CpuKernelBackend.Managed, 4);

        var expected = reference.EvaluatePromptLogits(ReferenceBenchmarkFixture.PromptTokens);
        var actual = managed.EvaluatePromptLogits(ReferenceBenchmarkFixture.PromptTokens);
        var largestError = expected.Zip(actual, (left, right) => Math.Abs(left - right)).Max();
        var range = expected.Max() - expected.Min();

        await Assert.That(ArgMax(actual)).IsEqualTo(ArgMax(expected));
        await Assert.That(largestError).IsLessThanOrEqualTo(range * 0.05f);
    }

    private static Qwen2Model Load(CpuKernelBackend backend, int threads, int? prefillChunkTokens = null)
    {
        var options = new ModelLoadOptions
        {
            ContextSize = 64,
            MaximumParallelism = threads,
            KernelBackend = backend,
        };
        if (prefillChunkTokens is { } chunk)
        {
            options = options with { PrefillChunkTokens = chunk };
        }

        return (Qwen2Model)ModelLoader.Load(ReferenceBenchmarkFixture.GetModelPath(), options);
    }

    private static int[] Bits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];

    private static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());
}
