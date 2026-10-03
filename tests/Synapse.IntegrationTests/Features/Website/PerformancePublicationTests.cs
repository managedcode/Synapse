using System.Diagnostics;
using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Website;

public sealed class PerformancePublicationTests
{
    [Test]
    public async Task PerformancePublicationExportsPartialMeasuredEvidence()
    {
        using var fixture = new PublicationFixture();
        fixture.CopySmoke();

        var (exit, error) = await fixture.RunAsync();

        await Assert.That(exit).IsEqualTo(3).Because(error);
        using var document = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.JsonPath));
        var result = document.RootElement;
        await Assert.That(result.GetProperty("schema_version").GetInt32()).IsEqualTo(1);
        await Assert.That(result.GetProperty("complete_artifacts").GetInt32()).IsEqualTo(1);
        await Assert.That(result.GetProperty("expected_artifacts").GetInt32()).IsEqualTo(22);
        await Assert.That(result.GetProperty("missing_artifacts").GetArrayLength()).IsEqualTo(21);
        await Assert.That(result.GetProperty("invalid_artifacts").GetArrayLength()).IsEqualTo(0);
        await Assert.That(result.GetProperty("generated_at_utc").GetDateTimeOffset().Offset)
            .IsEqualTo(TimeSpan.Zero);
        var rows = result.GetProperty("rows").EnumerateArray().ToArray();
        await Assert.That(rows.Length).IsEqualTo(4);
        var synapse = rows.Single(row => row.GetProperty("subject").GetString() == "synapse");
        await Assert.That(synapse.GetProperty("samples").GetInt32()).IsEqualTo(5);
        await Assert.That(synapse.GetProperty("output_tokens").GetDouble()).IsEqualTo(8);
        await Assert.That(synapse.GetProperty("source_artifact").GetString())
            .IsEqualTo("performance-osx-arm64");
        await Assert.That(synapse.GetProperty("output_state").GetString()).IsEqualTo("matched");
        await Assert.That(synapse.GetProperty("wall_metric_scope").GetString())
            .IsEqualTo("fresh_process");
        await Assert.That(synapse.GetProperty("wall_milliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(synapse.GetProperty("peak_rss_mib").GetDouble()).IsGreaterThan(0);
    }

    [Test]
    public async Task PerformancePublicationPreservesMissingAndNativePhaseMetrics()
    {
        using var fixture = new PublicationFixture();
        fixture.CopySmoke();

        var (exit, error) = await fixture.RunAsync();

        await Assert.That(exit).IsEqualTo(3).Because(error);
        using var document = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.JsonPath));
        var native = document.RootElement.GetProperty("rows").EnumerateArray()
            .Single(row => row.GetProperty("subject").GetString() == "llamacpp");
        await Assert.That(native.GetProperty("ttft_milliseconds").ValueKind)
            .IsEqualTo(JsonValueKind.Null);
        await Assert.That(native.GetProperty("decode_tokens_per_second").GetDouble())
            .IsGreaterThan(0);
        await Assert.That(native.GetProperty("decode_metric_scope").GetString())
            .IsEqualTo("native_eval");
        await Assert.That(native.GetProperty("cohort").GetString())
            .IsEqualTo("GGUF CPU, fresh process");
    }

    [Test]
    public async Task PerformancePublicationRejectsMislabeledArtifacts()
    {
        using var fixture = new PublicationFixture();
        fixture.CopyEvidence("foundry-local-ubuntu-24.04-qwen2.5-0.5b", "foundry-single.json",
            "2026-09-28-m2-pro-foundry-local-phi-4-mini-cpu-capitals-single-128-diagnostic.json");

        var (exit, error) = await fixture.RunAsync();

        await Assert.That(exit).IsEqualTo(3).Because(error);
        using var document = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.JsonPath));
        var result = document.RootElement;
        await Assert.That(result.GetProperty("rows").GetArrayLength()).IsEqualTo(0);
        await Assert.That(result.GetProperty("invalid_artifacts").GetArrayLength()).IsEqualTo(1);
        await Assert.That(result.GetProperty("invalid_artifacts")[0].GetString())
            .Contains("alias mismatch");
    }

    [Test]
    public async Task PerformancePublicationNeverOverwritesPublishedJson()
    {
        using var fixture = new PublicationFixture();
        fixture.CopySmoke();
        // language=json
        const string Published = "{\"published\":true}";
        await File.WriteAllTextAsync(fixture.JsonPath, Published);

        var (exit, error) = await fixture.RunAsync();

        await Assert.That(exit).IsEqualTo(1).Because(error);
        await Assert.That(await File.ReadAllTextAsync(fixture.JsonPath)).IsEqualTo(Published);
        await Assert.That(Directory.EnumerateFiles(fixture.DirectoryPath, "*.tmp").Any()).IsFalse();
    }

    private sealed class PublicationFixture : IDisposable
    {
        public string DirectoryPath { get; } = Path.Combine(Path.GetTempPath(),
            $"performance-publication-{Guid.NewGuid():N}");
        public string JsonPath => Path.Combine(DirectoryPath, "results.json");

        public PublicationFixture()
        {
            Directory.CreateDirectory(DirectoryPath);
        }

        public void CopySmoke() => CopyEvidence("performance-osx-arm64", "synapse-benchmark.json",
            "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json");

        public void CopyEvidence(string artifact, string name, string source)
        {
            var directory = Path.Combine(DirectoryPath, "artifacts", artifact);
            Directory.CreateDirectory(directory);
            File.Copy(Path.Combine(FindRepositoryRoot(), "benchmarks", "results", source),
                Path.Combine(directory, name));
        }

        public async Task<(int Exit, string Error)> RunAsync()
        {
            var start = new ProcessStartInfo("dotnet")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            foreach (var argument in new[]
            {
                Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"),
                "aggregate", "--artifacts", Path.Combine(DirectoryPath, "artifacts"),
                "--model-set", Path.Combine(FindRepositoryRoot(), "benchmarks", "model-sets",
                    "foundry-local-families.json"),
                "--output", Path.Combine(DirectoryPath, "summary.md"), "--json", JsonPath,
            })
            {
                start.ArgumentList.Add(argument);
            }

            using var process = Process.Start(start)
                ?? throw new InvalidOperationException("Publication reporter process did not start.");
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            try
            {
                await process.WaitForExitAsync(timeout.Token);
            }
            catch (OperationCanceledException)
            {
                process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync();
                throw;
            }

            _ = await output;
            return (process.ExitCode, await error);
        }

        public void Dispose() => Directory.Delete(DirectoryPath, recursive: true);
    }
}
