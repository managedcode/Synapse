using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>The context sweep report: three axes from measured samples only, and KV size from the model geometry.</summary>
public sealed class SweepReportTests
{
    [Test]
    public async Task SweepOptionsRejectIncompleteOrTooSmallContexts()
    {
        string[] complete =
        [
            "--model", "m.gguf", "--haystack", "h.txt", "--contexts", "4096,8192", "--output", "o.json",
            "--subjects", "synapse:metal:f16,llamacpp:metal:f16", "--synapse-executable", "synapse",
        ];

        await Assert.That(SweepOptions.Parse(complete)).IsNotNull();
        await Assert.That(SweepOptions.Parse(complete[..^2])).IsNull();
        await Assert.That(SweepOptions.Parse([.. complete, "--contexts", "256"])).IsNull();
        await Assert.That(SweepOptions.Parse([.. complete, "--measurements", "zero"])).IsNull();
    }

    [Test]
    public async Task SummaryUsesMeasuredMediansAndModelKvGeometry()
    {
        var options = SweepOptions.Parse(
        [
            "--model", "m.gguf", "--haystack", "h.txt", "--contexts", "32768", "--output", "o.json",
            "--subjects", "synapse:metal:f16", "--synapse-executable", "synapse", "--reference", "synapse-metal/f16",
        ])!;
        var model = new SweepModel(676_000_000, 24, 2, 64);
        SweepSample[] samples =
        [
            Sample(round: 0, warmup: true, ttft: 900, footprint: 9_000),
            Sample(round: 1, warmup: false, ttft: 100, footprint: 1_000),
            Sample(round: 2, warmup: false, ttft: 300, footprint: 3_000),
        ];

        var row = SweepReport.Summarize(options, model, Tokenizer(), samples).Single();

        await Assert.That(row.KvCacheMebibytes).IsEqualTo(384.0);
        await Assert.That(row.TimeToFirstTokenMilliseconds).IsEqualTo(200.0);
        await Assert.That(row.PeakFootprintMebibytes).IsEqualTo(2_000 / 1048576.0);
        await Assert.That(row.MeasuredRuns).IsEqualTo(2);
        await Assert.That(row.CorrectRuns).IsEqualTo(2);
        await Assert.That(row.TokensMatchingReference ?? 0).IsGreaterThan(0);
        await Assert.That(SweepReport.Markdown([row])).Contains("| 32768 | synapse-metal | f16 | 384 |");
    }

    [Test]
    public async Task MatchingPrefixCountsSharedLeadingTokens()
    {
        var tokenizer = Tokenizer();

        await Assert.That(SweepReport.MatchingPrefix(tokenizer, "The capital of France is Paris.", "The capital of France is Paris."))
            .IsEqualTo(tokenizer.Encode("The capital of France is Paris.", parseSpecialTokens: false).Count);
        await Assert.That(SweepReport.MatchingPrefix(tokenizer, "The capital of France is Paris.", "The capital of Spain is Madrid."))
            .IsEqualTo(tokenizer.Encode("The capital of", parseSpecialTokens: false).Count);
    }

    private static SweepSample Sample(int round, bool warmup, double ttft, long footprint) => new(
        32768, "synapse-metal", "f16", round, warmup, Correct: true,
        new SweepRun(0, "The number is 1234567. Summary follows.", 32_500, true, ttft, ttft + 1_000, 128, 90.0,
            ttft + 2_000, footprint, footprint, null));

    private static ITextTokenizer Tokenizer() => TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
}
