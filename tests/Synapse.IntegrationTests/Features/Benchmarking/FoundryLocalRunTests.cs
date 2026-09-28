using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.FoundryLocalFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel("foundry-local")]
public sealed class FoundryLocalRunTests
{
    private const string AnchorVariant = "qwen2.5-0.5b-instruct-generic-cpu:4";
    private const string RecordedEvidence =
        "2026-09-28-m2-pro-foundry-local-qwen2.5-0.5b-cpu-capitals-3turn-diagnostic.json";

    [Test]
    public async Task FoundryRunRefusesUncachedModelWithoutDownloading()
    {
        var cache = Directory.CreateTempSubdirectory("foundry-empty-cache-");
        var output = Path.Combine(cache.FullName, "evidence.json");
        try
        {
            var result = await RunAsync("run", "--set", ModelSetPath, "--alias", AnchorAlias,
                "--cache", cache.FullName, "--scenario", ScenarioPath("capitals-single-long.json"),
                "--output", output);

            await Assert.That(result.ExitCode).IsEqualTo(4).Because(result.StandardError);
            await Assert.That(result.StandardError).Contains($"'{AnchorVariant}' is not cached");
            await Assert.That(File.Exists(output)).IsFalse();
            await Assert.That(Directory.EnumerateFiles(cache.FullName, "*.onnx*", SearchOption.AllDirectories))
                .IsEmpty();
        }
        finally
        {
            cache.Delete(recursive: true);
        }
    }

    [Test]
    public async Task FoundryRunRecordsRealStreamingEvidence()
    {
        var output = Path.Combine(Path.GetTempPath(), $"foundry-evidence-{Guid.NewGuid():N}.json");
        try
        {
            var result = await RunAsync("run", "--set", ModelSetPath, "--alias", AnchorAlias,
                "--cache", CacheDirectory, "--scenario", ScenarioPath("capitals-france-us-uk-3-turns.json"),
                "--output", output, "--max-tokens", "8", "--warmups", "0", "--measurements", "1",
                "--runner-label", "tunit");

            await Assert.That(result.ExitCode).IsEqualTo(0).Because(
                $"{result.StandardError}\nFetch the anchor first: fetch --set <set> --alias {AnchorAlias} --cache {CacheDirectory}");
            using var evidence = JsonDocument.Parse(await File.ReadAllTextAsync(output));
            await AssertProvenanceAsync(evidence.RootElement);
            await AssertSamplesAsync(evidence.RootElement.GetProperty("samples"));
        }
        finally
        {
            File.Delete(output);
        }
    }

    [Test]
    public async Task FoundryReportRendersSeparateCohortFromRecordedEvidence()
    {
        var input = Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), "benchmarks", "results",
            RecordedEvidence);

        var result = await RunAsync("report", "--input", input);

        await Assert.That(result.ExitCode).IsEqualTo(0).Because(result.StandardError);
        await Assert.That(result.StandardOutput).Contains("Foundry Local");
        await Assert.That(result.StandardOutput).Contains(AnchorVariant);
        await Assert.That(result.StandardOutput).Contains("CPUExecutionProvider");
        await Assert.That(result.StandardOutput).Contains("|3|");
        await Assert.That(result.StandardOutput).Contains("Peak RSS MiB");
        await Assert.That(result.StandardOutput).Contains("Do not rank this cohort against the GGUF or MLX cohorts");
    }

    private static async Task AssertProvenanceAsync(JsonElement root)
    {
        await Assert.That(root.GetProperty("status").GetString())
            .IsEqualTo("measured_foundry_local_separate_onnx_cohort_quality_unreviewed");
        await Assert.That(root.GetProperty("variant_id").GetString()).IsEqualTo(AnchorVariant);
        await Assert.That(root.GetProperty("family").GetString()).IsEqualTo("qwen");
        await Assert.That(root.GetProperty("device").GetString()).IsEqualTo("cpu");
        await Assert.That(root.GetProperty("execution_provider").GetString()).IsEqualTo("CPUExecutionProvider");
        await Assert.That(root.GetProperty("sdk_version").GetString()).StartsWith("2.0.1");
        await Assert.That(root.GetProperty("thread_policy").GetString())
            .IsEqualTo("runtime_default_not_configurable");
        await Assert.That(root.GetProperty("context_mode").GetString())
            .Contains("fresh_chat_session_per_request");
        await Assert.That(root.GetProperty("runner_label").GetString()).IsEqualTo("tunit");
        await Assert.That(root.GetProperty("context_tokens").GetInt32()).IsEqualTo(1024);
        await Assert.That(root.GetProperty("original_max_length").GetInt32()).IsEqualTo(32768);
        await Assert.That(root.GetProperty("genai_search").GetProperty("max_length").GetInt32()).IsEqualTo(1024)
            .Because("ONNX Runtime GenAI preallocates KV for max_length, so the bound must be effective");
        await Assert.That(root.GetProperty("load_milliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("peak_resident_bytes").GetInt64()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("memory_sample_count").GetInt32()).IsGreaterThan(0);
        var files = root.GetProperty("model_files").EnumerateArray().ToArray();
        await Assert.That(files.Length).IsGreaterThan(0);
        foreach (var file in files)
        {
            await Assert.That(file.GetProperty("sha256").GetString()!.Length).IsEqualTo(64);
            await Assert.That(file.GetProperty("size_bytes").GetInt64()).IsGreaterThan(0);
        }
    }

    private static async Task AssertSamplesAsync(JsonElement samples)
    {
        await Assert.That(samples.GetArrayLength()).IsEqualTo(3);
        var previousPromptTokens = 0;
        foreach (var sample in samples.EnumerateArray())
        {
            var promptTokens = sample.GetProperty("prompt_tokens").GetInt32();
            var ttft = sample.GetProperty("time_to_first_token_milliseconds").GetDouble();
            await Assert.That(promptTokens).IsGreaterThan(previousPromptTokens)
                .Because("each turn resends the growing locked transcript");
            await Assert.That(sample.GetProperty("generated_tokens").GetInt32()).IsBetween(1, 8);
            await Assert.That(ttft).IsGreaterThan(0);
            await Assert.That(sample.GetProperty("request_wall_milliseconds").GetDouble())
                .IsGreaterThanOrEqualTo(ttft);
            await Assert.That(sample.GetProperty("process_cpu_milliseconds").GetDouble()).IsGreaterThan(0);
            await Assert.That(sample.GetProperty("quality_status").GetString()).IsEqualTo("unreviewed");
            previousPromptTokens = promptTokens;
        }
    }
}
