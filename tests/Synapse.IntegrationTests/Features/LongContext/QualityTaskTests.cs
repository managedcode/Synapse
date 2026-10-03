using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>The quality task generator (ADR-015): deterministic cases, answers present, budget respected.</summary>
public sealed class QualityTaskTests
{
    [Test]
    public async Task CasesAreDeterministicAndHoldTheirAnswers()
    {
        var factory = CreateFactory();

        foreach (var task in QualityTaskFactory.TaskNames)
        {
            var first = factory.Create(task, 3_000, 0.3, seed: 7);
            var again = factory.Create(task, 3_000, 0.3, seed: 7);
            var other = factory.Create(task, 3_000, 0.3, seed: 8);

            await Assert.That(again.Tokens).IsEquivalentTo(first.Tokens).Because(task);
            await Assert.That(other.Tokens.SequenceEqual(first.Tokens)).IsFalse().Because(task);
            await Assert.That(first.Tokens.Length).IsBetween(2_000, 3_000).Because(task);
            await Assert.That(first.Prompt).EndsWith("<|im_start|>assistant\n");
            foreach (var answer in first.Answers)
            {
                await Assert.That(first.UserMessage).Contains(answer).Because(task);
            }

            await Assert.That(first.Grade(string.Join(", ", first.Answers))).IsEqualTo((1.0, true));
            await Assert.That(first.Grade("I do not know.")).IsEqualTo((0.0, false));
        }
    }

    [Test]
    public async Task DepthControlsNeedlePlacement()
    {
        var factory = CreateFactory();

        var early = factory.Create("needle", 3_000, 0.1, seed: 11);
        var late = factory.Create("needle", 3_000, 0.9, seed: 11);

        var earlyAt = (double)early.UserMessage.IndexOf(early.Answers[0], StringComparison.Ordinal) / early.UserMessage.Length;
        var lateAt = (double)late.UserMessage.IndexOf(late.Answers[0], StringComparison.Ordinal) / late.UserMessage.Length;
        await Assert.That(earlyAt).IsLessThan(0.3);
        await Assert.That(lateAt).IsGreaterThan(0.7);
    }

    [Test]
    public async Task HaystackShorterThanTheBudgetFailsExplicitly()
    {
        var factory = new QualityTaskFactory(Tokenizer(), "One short paragraph.\n\nAnother one.");

        await Assert.That(() => factory.Create("needle", 3_000, 0.5, seed: 1)).Throws<InvalidDataException>();
    }

    [Test]
    [Arguments("needle")]
    [Arguments("multikey")]
    [Arguments("vartrack")]
    public async Task CorpusLineEndingsDoNotChangeQualityCases(string task)
    {
        var docs = Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), "docs");
        var haystack = string.Join("\n\n", Directory.GetFiles(docs, "*.md", SearchOption.AllDirectories)
            .Order(StringComparer.Ordinal).Select(File.ReadAllText)).ReplaceLineEndings("\n");
        var tokenizer = Tokenizer();
        var expected = new QualityTaskFactory(tokenizer, haystack).Create(task, 3_000, 0.3, 7);
        var windows = new QualityTaskFactory(tokenizer, haystack.ReplaceLineEndings("\r\n"))
            .Create(task, 3_000, 0.3, 7);
        await Assert.That(windows.Prompt).IsEqualTo(expected.Prompt);
        await Assert.That(windows.Tokens).IsEquivalentTo(expected.Tokens);
        await Assert.That(windows.Tokens.Length).IsBetween(2_000, 3_000);
    }

    private static QualityTaskFactory CreateFactory()
    {
        var docs = Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), "docs");
        var haystack = string.Join("\n\n", Directory.GetFiles(docs, "*.md", SearchOption.AllDirectories)
            .Order(StringComparer.Ordinal)
            .Select(File.ReadAllText));
        return new QualityTaskFactory(Tokenizer(), haystack);
    }

    private static ITextTokenizer Tokenizer() => TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
}
