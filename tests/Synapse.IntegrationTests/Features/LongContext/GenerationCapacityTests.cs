using System.Text.Json;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>TEST-CTX-009-1: command capacity follows actual generation work, with real allocation and output evidence.</summary>
[NotInParallel]
public sealed class GenerationCapacityTests
{
    private const int Vocabulary = 131_072;
    private static readonly int[] Prompt = [1, 2, 3, 4];

    [Test]
    public async Task SingleGenerationAvoidsUnusedVocabularyRowsWithoutChangingOutput()
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync(new(1, 1, 1, 64, 64, Vocabulary, 64));
        _ = Measure(fixture.Model, rows: 1);
        var (narrowBytes, narrowKvBytes, narrowTokens) = Measure(fixture.Model, rows: 1);
        var (wideBytes, _, wideTokens) = Measure(fixture.Model, rows: 8);
        var (exit, output, error) = await CliScoreTests.RunCliAsync("generate", "--model", fixture.Model,
            "--tokens", "1,2,3,4", "--max-tokens", "2", "--context-size", "64", "--threads", "1",
            "--backend", "managed");

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        var actual = json.RootElement.GetProperty("managed_allocated_during_subject_bytes").GetInt64();
        Console.WriteLine($"generation capacity: one-row={narrowBytes}, eight-row={wideBytes}, CLI={actual} allocated bytes");
        await Assert.That(wideTokens.SequenceEqual(narrowTokens)).IsTrue();
        await Assert.That(json.RootElement.GetProperty("generated_tokens").EnumerateArray().Select(id => id.GetInt32())
            .SequenceEqual(narrowTokens)).IsTrue();
        await Assert.That(wideBytes - narrowBytes).IsGreaterThanOrEqualTo(6L * Vocabulary * sizeof(float));
        await Assert.That(actual).IsLessThan(wideBytes - (2 * 1_024 * 1_024));
        await Assert.That(json.RootElement.GetProperty("allocated_kv_bytes").GetInt64()).IsEqualTo(narrowKvBytes);
    }

    [Test]
    [Arguments(1)]
    [Arguments(7)]
    public async Task SpeculationRetainsEveryRequiredVerificationRow(int draftTokens)
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync();
        using var target = Qwen2Model.Load(fixture.Model, new ModelLoadOptions
        {
            ContextSize = 64,
            MaximumParallelism = 1,
            MaximumConcurrentSessions = 1,
            ScoringRowsPerStep = 1,
        });
        var expected = target.Generate(Prompt, 16);
        var (exit, output, error) = await CliScoreTests.RunCliAsync("generate", "--model", fixture.Model,
            "--tokens", "1,2,3,4", "--max-tokens", "16", "--context-size", "64", "--threads", "1",
            "--draft-model", fixture.Model, "--draft-tokens", draftTokens.ToString(System.Globalization.CultureInfo.InvariantCulture));

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("generated_tokens").EnumerateArray().Select(id => id.GetInt32())
            .SequenceEqual(expected.GeneratedTokens)).IsTrue();
        await Assert.That(json.RootElement.GetProperty("speculation").GetProperty("draft_tokens").GetInt32()).IsEqualTo(draftTokens);
    }

    [Test]
    [Arguments("0")]
    [Arguments("8")]
    [Arguments("2147483647")]
    public async Task DraftCountOutsideItsDocumentedBoundsFailsBeforeLoading(string count) =>
        await Assert.That(GenerationOptions.Parse(["--model", "unused.synapse", "--tokens", "1",
            "--draft-model", "unused.synapse", "--draft-tokens", count])).IsNull();

    private static (long Bytes, long KvBytes, IReadOnlyList<int> Tokens) Measure(string path, int rows)
    {
        var before = GC.GetTotalAllocatedBytes(precise: true);
        using var model = Qwen2Model.Load(path, new ModelLoadOptions
        {
            ContextSize = 64,
            MaximumParallelism = 1,
            MaximumConcurrentSessions = rows == 1 ? 1 : 4,
            ScoringRowsPerStep = rows,
        });
        var generated = model.Generate(Prompt, 2).GeneratedTokens;
        return (GC.GetTotalAllocatedBytes(precise: true) - before, model.AllocatedKvBytes, generated);
    }
}
