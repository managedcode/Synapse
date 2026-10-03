using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class BenchmarkingInputContractTests
{
    private static readonly string[] Scenarios =
    [
        "capitals-single-long.json", "capitals-france-us-uk-3-turns.json", "passkey-qwen2.5.json",
        "travel-planner-10-turns.json", "embedding-retrieval-small.json",
    ];

    private static readonly string[] RecordedFixtures =
    [
        "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-four-subject-memory-clr-smoke.json",
        "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-32tok-quality-divergence-final.json",
        "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json",
        "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-3turn-diagnostic.json",
        "2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-3turn-diagnostic.json",
        "2026-09-28-m2-pro-foundry-local-qwen2.5-0.5b-cpu-capitals-3turn-diagnostic.json",
        "2026-09-28-m2-pro-foundry-local-phi-4-mini-cpu-capitals-single-128-diagnostic.json",
    ];

    [Test]
    public async Task BenchmarkInputsRemainAvailableOutsideIgnoredResultsDirectory()
    {
        var ignore = await File.ReadAllLinesAsync(Path.Combine(FindRepositoryRoot(), ".gitignore"));
        await Assert.That(ignore.Any(line => line.TrimEnd('/') is "benchmarks" or "/benchmarks"))
            .IsTrue();
        var result = await FoundryLocalFixture.RunAsync("plan", "--set",
            BenchmarkInputPath("ModelSets", "foundry-local-families.json"));
        await Assert.That(result.ExitCode).IsEqualTo(0).Because(result.StandardError);
        using var matrix = JsonDocument.Parse(result.StandardOutput);
        await Assert.That(matrix.RootElement.GetProperty("include").GetArrayLength()).IsGreaterThan(0);
        foreach (var scenario in Scenarios)
        {
            using var document = JsonDocument.Parse(await File.ReadAllTextAsync(
                BenchmarkInputPath("Scenarios", scenario)));
            await Assert.That(document.RootElement.GetProperty("schemaVersion").GetInt32()).IsEqualTo(1);
            await Assert.That(document.RootElement.GetProperty("id").GetString())
                .IsEqualTo(Path.GetFileNameWithoutExtension(scenario));
        }

        foreach (var fixture in RecordedFixtures)
        {
            using var document = JsonDocument.Parse(await File.ReadAllTextAsync(RecordedBenchmarkPath(fixture)));
            await Assert.That(document.RootElement.ValueKind).IsEqualTo(JsonValueKind.Object);
            await Assert.That(document.RootElement.GetProperty("samples").GetArrayLength()).IsGreaterThan(0);
        }
    }
}
