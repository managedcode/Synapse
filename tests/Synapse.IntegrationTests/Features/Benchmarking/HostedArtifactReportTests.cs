using System.Diagnostics;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class HostedArtifactReportTests
{
    [Test]
    public async Task HostedReportShowsPartialEvidenceAndMissingArtifacts()
    {
        var root = FindRepositoryRoot();
        var temporary = Path.Combine(Path.GetTempPath(), $"hosted-report-{Guid.NewGuid():N}");
        try
        {
            var artifact = Path.Combine(temporary, "performance-osx-arm64");
            Directory.CreateDirectory(artifact);
            File.Copy(Path.Combine(root, "benchmarks", "results",
                "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json"),
                Path.Combine(artifact, "synapse-benchmark.json"));

            var (exit, error) = await RunReportAsync(root, temporary);

            await Assert.That(exit).IsEqualTo(3).Because(error);
            var report = await File.ReadAllTextAsync(Path.Combine(temporary, "summary.md"));
            await Assert.That(report).Contains("1/22 expected raw artifacts");
            await Assert.That(report).Contains("|8-token smoke|");
            await Assert.That(report).Contains("|synapse|performance-osx-arm64|5|8|");
            await Assert.That(report).Contains("Missing artifacts (21)");
            await Assert.That(report).Contains("Long-output quality is unreviewed");
        }
        finally
        {
            if (Directory.Exists(temporary))
            {
                Directory.Delete(temporary, recursive: true);
            }
        }
    }

    [Test]
    public async Task HostedReportRejectsMislabeledEvidence()
    {
        var root = FindRepositoryRoot();
        var temporary = Path.Combine(Path.GetTempPath(), $"hosted-report-{Guid.NewGuid():N}");
        try
        {
            var artifact = Path.Combine(temporary, "foundry-local-ubuntu-24.04-qwen2.5-0.5b");
            Directory.CreateDirectory(artifact);
            File.Copy(Path.Combine(root, "benchmarks", "results",
                "2026-09-28-m2-pro-foundry-local-phi-4-mini-cpu-capitals-single-128-diagnostic.json"),
                Path.Combine(artifact, "foundry-single.json"));

            var (exit, error) = await RunReportAsync(root, temporary);

            await Assert.That(exit).IsEqualTo(3).Because(error);
            var report = await File.ReadAllTextAsync(Path.Combine(temporary, "summary.md"));
            await Assert.That(report).Contains("Invalid artifacts (1)");
            await Assert.That(report).Contains("alias mismatch");
            await Assert.That(report).DoesNotContain("|phi-4-mini|");
        }
        finally
        {
            if (Directory.Exists(temporary))
            {
                Directory.Delete(temporary, recursive: true);
            }
        }
    }

    private static async Task<(int ExitCode, string StandardError)> RunReportAsync(string root, string temporary)
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
            "aggregate", "--artifacts", temporary,
            "--model-set", Path.Combine(root, "benchmarks", "model-sets", "foundry-local-families.json"),
            "--output", Path.Combine(temporary, "summary.md"),
        })
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Hosted report process did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
        _ = await output;
        return (process.ExitCode, await error);
    }
}
