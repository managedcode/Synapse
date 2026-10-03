using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Dynamic KV capacity (ADR-017): memory follows the context in use; growth never changes the numbers.</summary>
[NotInParallel]
public sealed class DynamicKvCapacityTests
{
    private static readonly int[] Prompt = [.. Enumerable.Range(0, 700).Select(index => (index * 7919 % 150_000) + 100)];
    private static readonly int[] Continuation = [.. Enumerable.Range(0, 40).Select(index => (index * 104_729 % 150_000) + 100)];

    [Test]
    public async Task CpuKvCacheGrowsAndKeepsContents()
    {
        var cache = new Qwen2KvCache(layers: 2, maximumPositions: 8_192, kvWidth: 4, growthPositions: 1_024);
        float[] key = [1, 2, 3, 4];
        float[] value = [5, 6, 7, 8];

        cache.Store(1, 5, key, value);
        var first = cache.AllocatedPositions;
        for (var position = 0; position < 1_500; position++)
        {
            cache.Store(0, position, [position, 0, 0, 0], [0, 0, 0, position]);
        }

        await Assert.That(first).IsEqualTo(1_024);
        await Assert.That(cache.AllocatedPositions).IsEqualTo(2_048);
        await Assert.That(cache.GetKey(1, 5, 0, 4).ToArray()).IsEquivalentTo(key);
        await Assert.That(cache.GetValue(1, 5, 0, 4).ToArray()).IsEquivalentTo(value);
        await Assert.That(cache.GetKey(0, 1_499, 0, 4)[0]).IsEqualTo(1_499f);
        await Assert.That(() => cache.Store(0, 8_192, key, value)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Metal)]
    public async Task GrowingKvSlotMatchesFullSlotBitwise(KernelBackend backend)
    {
        if (backend == KernelBackend.Metal)
        {
            GpuHardware.RequireMetal();
        }

        using var growing = Load(backend, contextSize: 2_048, growthPositions: 64);
        using var full = Load(backend, contextSize: 2_048, growthPositions: 2_048);

        var expected = full.EvaluateIncrementalLogits(Prompt, Continuation);
        var actual = growing.EvaluateIncrementalLogits(Prompt, Continuation);

        await Assert.That(actual).IsEquivalentTo(expected);
        await Assert.That(growing.AllocatedKvBytes).IsLessThan(full.AllocatedKvBytes);
    }

    [Test]
    [Arguments(KernelBackend.Managed, 24_576)]
    [Arguments(KernelBackend.Metal, 24_576)]
    public async Task KvBytesFollowTheContextInUse(KernelBackend backend, int bytesPerPosition)
    {
        if (backend == KernelBackend.Metal)
        {
            GpuHardware.RequireMetal();
        }

        using var model = Load(backend, contextSize: 32_768, growthPositions: 1_024);

        _ = model.EvaluatePromptLogits([.. Prompt.Take(100)]);
        var small = model.AllocatedKvBytes;
        _ = model.EvaluatePromptLogits([.. Enumerable.Repeat(Prompt, 3).SelectMany(tokens => tokens).Take(1_500)]);

        // Sequential writes grow 1,024 to 2,048 positions: the need rounded up, at least doubling (ADR-017).
        await Assert.That(small).IsEqualTo(1_024L * bytesPerPosition);
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(2_048L * bytesPerPosition);
    }

    [Test]
    [Arguments(KernelBackend.Managed, 24_576)]
    [Arguments(KernelBackend.Metal, 24_576)]
    public async Task GenerateReservesPromptAndOutputOnce(KernelBackend backend, int bytesPerPosition)
    {
        if (backend == KernelBackend.Metal)
        {
            GpuHardware.RequireMetal();
        }

        using var model = Load(backend, contextSize: 8_192, growthPositions: 1_024);
        int[] prompt = [.. Enumerable.Repeat(Prompt, 3).SelectMany(tokens => tokens).Take(1_500)];

        _ = model.Generate(prompt, 600);

        // Growing step by step would end at 4,096 positions (1,024 → 2,048 → 4,096); one reservation holds 3,072.
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(3_072L * bytesPerPosition);
    }

    private static Qwen2Model Load(KernelBackend backend, int contextSize, int growthPositions) => (Qwen2Model)ModelLoader.LoadSourceForValidation(
        ReferenceBenchmarkFixture.GetModelPath(),
        new ModelLoadOptions
        {
            ContextSize = contextSize,
            MaximumParallelism = 8,
            KernelBackend = backend,
            KvGrowthPositions = growthPositions,
        });
}
