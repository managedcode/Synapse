using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Query-aware KV page activation (ADR-016): selection semantics, the dense boundary, and explicit scope.</summary>
[NotInParallel]
public sealed class KvPageActivationTests
{
    private const int Dimension = 64;

    [Test]
    public async Task KeyBoundNeverUnderestimatesAPageScore()
    {
        var random = new Random(5);
        var keys = Values(random, 640 * Dimension);
        var queries = Values(random, 3 * Dimension);

        for (var page = 0; page < 10; page++)
        {
            var bound = KvPageSelector.PageBound(queries, 3, Dimension, position => keys.AsSpan(position * Dimension, Dimension),
                page * 64, (page * 64) + 63);
            var exact = float.NegativeInfinity;
            for (var head = 0; head < 3; head++)
            {
                for (var position = page * 64; position < (page + 1) * 64; position++)
                {
                    exact = MathF.Max(exact, Dot(queries.AsSpan(head * Dimension, Dimension), keys.AsSpan(position * Dimension, Dimension)));
                }
            }

            await Assert.That(bound).IsGreaterThanOrEqualTo(exact - 1e-4f);
        }
    }

    [Test]
    public async Task SelectionKeepsSinkWindowAndTheBestPage()
    {
        var random = new Random(7);
        var keys = Values(random, 640 * Dimension, scale: 0.01f);
        var query = Values(random, Dimension);
        for (var position = 5 * 64; position < 6 * 64; position++)
        {
            query.AsSpan().CopyTo(keys.AsSpan(position * Dimension, Dimension));
        }

        var selected = new bool[640];
        KvPageSelector.Select(new KvPageActivation(1, 64), query, 1, Dimension,
            position => keys.AsSpan(position * Dimension, Dimension), 639, seed: 0, selected);

        await Assert.That(Pages(selected)).IsEquivalentTo([0, 5, 9]);
    }

    [Test]
    public async Task SmallerPagesSelectAtTheirOwnGranularity()
    {
        var random = new Random(8);
        var keys = Values(random, 640 * Dimension, scale: 0.01f);
        var query = Values(random, Dimension);
        for (var position = 208; position < 224; position++)
        {
            query.AsSpan().CopyTo(keys.AsSpan(position * Dimension, Dimension));
        }

        var selected = new bool[640];
        var activation = new KvPageActivation(1, 64) { PageTokens = 16 };
        KvPageSelector.Select(activation, query, 1, Dimension, position => keys.AsSpan(position * Dimension, Dimension),
            639, seed: 0, selected);

        await Assert.That(activation.Name).IsEqualTo("kvpages1w64p16");
        await Assert.That(Enumerable.Range(0, 640).Where(position => selected[position]).ToArray())
            .IsEquivalentTo([.. Enumerable.Range(0, 16), .. Enumerable.Range(208, 16), .. Enumerable.Range(576, 64)]);
        await Assert.That(() => new KvPageActivation(1, 64) { PageTokens = 24 }.Validate()).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task RandomControlSelectsTheBudgetDeterministically()
    {
        var keys = Values(new Random(9), 1_280 * Dimension);
        var query = Values(new Random(10), Dimension);
        var activation = new KvPageActivation(3, 64) { Selection = KvPageSelection.Random };
        var first = new bool[1_280];
        var again = new bool[1_280];

        KvPageSelector.Select(activation, query, 1, Dimension, position => keys.AsSpan(position * Dimension, Dimension),
            1_279, seed: 42, first);
        KvPageSelector.Select(activation, query, 1, Dimension, position => keys.AsSpan(position * Dimension, Dimension),
            1_279, seed: 42, again);

        await Assert.That(again).IsEquivalentTo(first);
        await Assert.That(Pages(first).Length).IsEqualTo(1 + 3 + 1);
        await Assert.That(Pages(first)).Contains(0).And.Contains(19);
    }

    [Test]
    public async Task KvPagesCoveringThePrefixEqualDense()
    {
        int[] prompt = [.. Enumerable.Range(0, 1_000).Select(index => (index * 7919 % 150_000) + 100)];
        int[] continuation = [12095, 13, 1084, 374];
        using var dense = Load(activation: null);
        using var covering = Load(new KvPageActivation(64, 64));
        using var sparse = Load(new KvPageActivation(2, 128));

        var expected = dense.EvaluateIncrementalLogits(prompt, continuation);
        var actual = covering.EvaluateIncrementalLogits(prompt, continuation);
        var approximate = sparse.EvaluateIncrementalLogits(prompt, continuation);

        await Assert.That(actual).IsEquivalentTo(expected);
        await Assert.That(approximate.SequenceEqual(expected)).IsFalse();
    }

    [Test]
    public async Task KvPagesAreNamedAndRejectedWhereNotImplemented()
    {
        using var named = Load(new KvPageActivation(2, 128));
        using var random = Load(new KvPageActivation(2, 128) { Selection = KvPageSelection.Random });

        await Assert.That(named.RuntimeProfile).EndsWith("+kvpages2w128");
        await Assert.That(random.RuntimeProfile).EndsWith("+kvpages2w128r");
        foreach (var backend in new[] { KernelBackend.Reference, KernelBackend.Metal })
        {
            var exception = await Assert.That(() => ModelLoader.Load(ReferenceBenchmarkFixture.GetModelPath(),
                    new ModelLoadOptions { ContextSize = 64, KernelBackend = backend, KvPageActivation = new KvPageActivation(2, 64) }))
                .Throws<NotSupportedException>();
            await Assert.That(exception!.Message).Contains("ADR-016");
        }

        await Assert.That(() => new KvPageActivation(-1, 64).Validate()).Throws<ArgumentOutOfRangeException>();
    }

    private static Qwen2Model Load(KvPageActivation? activation) => (Qwen2Model)ModelLoader.Load(
        ReferenceBenchmarkFixture.GetModelPath(),
        new ModelLoadOptions
        {
            ContextSize = 1_100,
            MaximumParallelism = 8,
            KernelBackend = KernelBackend.Managed,
            KvPageActivation = activation,
        });

    private static float[] Values(Random random, int count, float scale = 1) =>
        [.. Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() * 2) - 1) * scale)];

    private static int[] Pages(bool[] selected) =>
        [.. Enumerable.Range(0, (selected.Length + 63) / 64).Where(page => selected.Skip(page * 64).Take(64).Any(value => value))];

    private static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        var sum = 0f;
        for (var index = 0; index < left.Length; index++)
        {
            sum += left[index] * right[index];
        }

        return sum;
    }
}
