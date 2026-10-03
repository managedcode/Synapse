using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Teacher-forced scoring (ADR-015): per-position NLL and greedy token against independent prefills.</summary>
[NotInParallel]
public sealed class TokenScoringTests
{
    internal const string Text =
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. " +
        "The capital of Spain is Madrid. The capital of Ukraine is Kyiv. The capital of Poland is Warsaw.";

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    public async Task ScoreMatchesIndependentPrefills(KernelBackend backend)
    {
        var tokens = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath()).Encode(Text);
        using var model = Load(backend, scoringRows: 3);
        const int First = 5;

        var scores = model.Score(tokens, First, progress: null);

        await Assert.That(scores.FirstScoredPosition).IsEqualTo(First);
        await Assert.That(scores.NegativeLogLikelihoods.Count).IsEqualTo(tokens.Count - 1 - First);
        for (var index = 0; index < scores.NegativeLogLikelihoods.Count; index++)
        {
            var position = First + index;
            var logits = model.EvaluatePromptLogits([.. tokens.Take(position + 1)]);
            await Assert.That(scores.GreedyTokens[index]).IsEqualTo(TokenScoring.ArgMax(logits));
            await Assert.That(Math.Abs(scores.NegativeLogLikelihoods[index] - TokenScoring.NegativeLogLikelihood(logits, tokens[position + 1])))
                .IsLessThanOrEqualTo(0.02);
        }
    }

    [Test]
    public async Task ScoreIsBackendConsistentAndKnowsCapitals()
    {
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
        var tokens = tokenizer.Encode(Text);
        using var reference = Load(KernelBackend.Reference, scoringRows: 8);
        using var native = Load(KernelBackend.Native, scoringRows: 8);

        var expected = reference.Score(tokens, 0, progress: null);
        var actual = native.Score(tokens, 0, progress: null);

        // Q8_0 activations (native profile) against FP32 activations (reference) measured 0.011 nats on this text.
        await Assert.That(actual.GreedyTokens).IsEquivalentTo(expected.GreedyTokens);
        await Assert.That(Math.Abs(actual.NegativeLogLikelihoods.Average() - expected.NegativeLogLikelihoods.Average()))
            .IsLessThanOrEqualTo(0.02);
        var kyiv = tokens.Count - 1 - tokenizer.Encode(" Kyiv. The capital of Poland is Warsaw.").Count;
        await Assert.That(tokenizer.Decode([expected.GreedyTokens[kyiv]])).IsIn(" Ky", " Kiev");
    }

    [Test]
    public async Task ScoringArgumentsAreValidated()
    {
        await Assert.That(() => Load(KernelBackend.Managed, scoringRows: 0)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => Load(KernelBackend.Managed, scoringRows: 513)).Throws<ArgumentOutOfRangeException>();
        using var model = Load(KernelBackend.Managed, scoringRows: 8);

        await Assert.That(() => model.Score([785, 6722], 1, progress: null)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => model.Score([785, 6722, 315], -1, progress: null)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => model.Score([.. Enumerable.Repeat(785, 65)], 0, progress: null)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task ScoreReportsProgressThroughEveryPosition()
    {
        var tokens = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath()).Encode(Text);
        using var model = Load(KernelBackend.Managed, scoringRows: 4);
        var reports = new List<GenerationProgress>();

        _ = model.Score(tokens, 20, new SynchronousProgress(reports.Add));

        await Assert.That(reports.Count).IsGreaterThan(1);
        await Assert.That(reports[^1].EvaluatedPromptTokens).IsEqualTo(tokens.Count - 1);
        await Assert.That(reports[^1].PromptTokens).IsEqualTo(tokens.Count - 1);
    }

    internal static Qwen2Model Load(KernelBackend backend, int scoringRows) => (Qwen2Model)ModelLoader.LoadSourceForValidation(
        ReferenceBenchmarkFixture.GetModelPath(),
        new ModelLoadOptions
        {
            ContextSize = 64,
            MaximumParallelism = 8,
            KernelBackend = backend,
            ScoringRowsPerStep = scoringRows,
        });

    private sealed class SynchronousProgress(Action<GenerationProgress> report) : IProgress<GenerationProgress>
    {
        public void Report(GenerationProgress value) => report(value);
    }
}
