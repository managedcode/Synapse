using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

[NotInParallel]
public sealed class KvCacheParityTests
{
    private static readonly int[] Prompt = [785, 6722, 315, 9625, 374];
    private static readonly int[] Continuation = [12095, 13, 1084, 374, 279, 7772];

    [Test]
    [Arguments(CpuKernelBackend.Reference)]
    [Arguments(CpuKernelBackend.Managed)]
    [Arguments(CpuKernelBackend.Native)]
    public async Task IncrementalDecodeMatchesFullPrefill(CpuKernelBackend backend)
    {
        using var model = Load(backend, contextSize: 64);

        var incremental = model.EvaluateIncrementalLogits(Prompt, Continuation);
        var full = model.EvaluatePromptLogits([.. Prompt, .. Continuation]);

        await Assert.That(Bits(incremental)).IsEquivalentTo(Bits(full));
    }

    [Test]
    [Arguments(CpuKernelBackend.Managed)]
    [Arguments(CpuKernelBackend.Native)]
    public async Task GenerationRepeatsAfterLongerPrompt(CpuKernelBackend backend)
    {
        using var model = Load(backend, contextSize: 128);
        int[] longer = [.. Enumerable.Repeat(Prompt, 8).SelectMany(tokens => tokens)];

        var first = model.Generate(Prompt, 6);
        _ = model.Generate(longer, 6);
        var again = model.Generate(Prompt, 6);

        await Assert.That(again.GeneratedTokens).IsEquivalentTo(first.GeneratedTokens);
    }

    [Test]
    public async Task LongPromptAcrossChunksMatchesSequential()
    {
        int[] prompt = [.. Enumerable.Range(0, 150).Select(index => Prompt[index % Prompt.Length] + (index / 7))];
        using var sequential = Load(CpuKernelBackend.Managed, contextSize: 160, prefillChunkTokens: 1);
        using var chunked = Load(CpuKernelBackend.Managed, contextSize: 160);

        var expected = sequential.EvaluatePromptLogits(prompt);
        var actual = chunked.EvaluatePromptLogits(prompt);

        await Assert.That(Bits(actual)).IsEquivalentTo(Bits(expected));
    }

    [Test]
    [Arguments(CpuKernelBackend.Reference)]
    [Arguments(CpuKernelBackend.Managed)]
    public async Task ContextCapacityIsEnforcedExactly(CpuKernelBackend backend)
    {
        using var model = Load(backend, contextSize: 8);

        var exact = model.Generate(Prompt, 3);

        await Assert.That(exact.GeneratedTokens.Count).IsEqualTo(3);
        await Assert.That(() => model.Generate(Prompt, 4)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(async () => await model.GenerateAsync(Prompt, 4, CancellationToken.None))
            .Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => model.Generate([.. Prompt, 999_999], 1)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task KvCacheKeepsHeadsSeparateAndRejectsOutOfRangePositions()
    {
        var cache = new Qwen2KvCache(layers: 2, capacity: 3, kvWidth: 8);
        float[] key = [0, 1, 2, 3, 4, 5, 6, 7];
        float[] value = [10, 11, 12, 13, 14, 15, 16, 17];

        cache.Store(1, 2, key, value);

        await Assert.That(cache.GetKey(1, 2, head: 1, headDimension: 4).ToArray()).IsEquivalentTo(new float[] { 4, 5, 6, 7 });
        await Assert.That(cache.GetValue(1, 2, head: 0, headDimension: 4).ToArray()).IsEquivalentTo(new float[] { 10, 11, 12, 13 });
        await Assert.That(cache.GetKey(0, 2, head: 0, headDimension: 4).ToArray().All(item => item == 0)).IsTrue();
        await Assert.That(() => cache.Store(1, 3, key, value)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => cache.Store(0, 0, key.AsSpan(0, 4), value)).Throws<ArgumentOutOfRangeException>();
    }

    private static Qwen2Model Load(CpuKernelBackend backend, int contextSize, int? prefillChunkTokens = null)
    {
        var options = new ModelLoadOptions { ContextSize = contextSize, MaximumParallelism = 3, KernelBackend = backend };
        if (prefillChunkTokens is { } chunk)
        {
            options = options with { PrefillChunkTokens = chunk };
        }

        return (Qwen2Model)ModelLoader.Load(ReferenceBenchmarkFixture.GetModelPath(), options);
    }

    private static int[] Bits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];
}
