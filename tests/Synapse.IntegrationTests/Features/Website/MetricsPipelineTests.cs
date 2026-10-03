using System.Diagnostics;
using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Website;

public sealed class MetricsPipelineTests
{
    [Test]
    public async Task SiteShellKeepsRecordedAndLiveMetricsSeparate()
    {
        var root = FindRepositoryRoot();
        var html = await File.ReadAllTextAsync(Path.Combine(root, "site", "index.html"));
        await Assert.That(html).Contains("id=\"metrics\"");
        await Assert.That(html).Contains("Recorded local measurements");
        await Assert.That(html).Contains("src=\"metrics.js\"");
        var script = await File.ReadAllTextAsync(Path.Combine(root, "site", "metrics.js"));
        await Assert.That(script).Contains("data/latest.json");
        await Assert.That(script).Contains("textContent");
        await Assert.That(script).DoesNotContain("innerHTML");
    }

    [Test]
    public async Task MetricsWorkflowsConnectTrustedRunsAndImmutableJson()
    {
        var root = FindRepositoryRoot();
        var pages = await File.ReadAllTextAsync(Path.Combine(root, ".github/workflows/pages.yml"));
        await Assert.That(pages).Contains("workflow_run:");
        await Assert.That(pages).Contains("workflows: [verify, performance]");
        await Assert.That(pages).Contains("github.event.workflow_run.head_branch == 'main'");
        await Assert.That(pages).Contains("github.event.workflow_run.event == 'push'");
        await Assert.That(pages).Contains("actions: read");
        await Assert.That(pages).Contains("collect-site-results.sh");
        await Assert.That(pages).Contains("site-data");
        var verify = await File.ReadAllTextAsync(Path.Combine(root, ".github/workflows/verify.yml"));
        await Assert.That(verify).Contains("--report-trx");
        await Assert.That(verify).Contains("--minimum-expected-tests 1");
        await Assert.That(verify).Contains("test-report");
        await Assert.That(verify).Contains("test-results-${{ matrix.runtime_identifier }}");
        var performance = await File.ReadAllTextAsync(Path.Combine(root, ".github/workflows/performance.yml"));
        await Assert.That(performance).Contains("--json \"${RUNNER_TEMP}/performance-results.json\"");
        var collector = await File.ReadAllTextAsync(Path.Combine(root, ".github/scripts/collect-site-results.sh"));
        await Assert.That(collector).Contains("run_attempt");
        await Assert.That(collector).Contains("head_branch == \"main\"");
        await Assert.That(collector).Contains("artifacts/${artifact_id}/zip");
    }

    [Test]
    public async Task SiteDataPublishesMissingEvidenceWithoutPassingCounts()
    {
        var temporary = Path.Combine(Path.GetTempPath(), $"site-data-{Guid.NewGuid():N}");
        Directory.CreateDirectory(temporary);
        try
        {
            var runs = Path.Combine(temporary, "runs.json");
            // language=json
            await File.WriteAllTextAsync(runs, "{\"verification\":null,\"performance\":null}");
            var output = Path.Combine(temporary, "latest.json");
            var first = await RunSiteDataAsync(temporary, runs, output);
            await Assert.That(first).IsEqualTo(0);
            using var json = JsonDocument.Parse(await File.ReadAllTextAsync(output));
            await Assert.That(json.RootElement.GetProperty("schema_version").GetInt32()).IsEqualTo(1);
            await Assert.That(json.RootElement.GetProperty("verification").GetProperty("reports")
                .GetArrayLength()).IsEqualTo(0);
            await Assert.That(json.RootElement.GetProperty("verification").GetProperty("missing_artifacts")
                .GetArrayLength()).IsEqualTo(3);
            await Assert.That(json.RootElement.GetProperty("performance").GetProperty("report")
                .ValueKind).IsEqualTo(JsonValueKind.Null);
            var before = await File.ReadAllBytesAsync(output);
            await Assert.That(await RunSiteDataAsync(temporary, runs, output)).IsEqualTo(1);
            await Assert.That(await File.ReadAllBytesAsync(output)).IsEquivalentTo(before);
        }
        finally
        {
            Directory.Delete(temporary, recursive: true);
        }
    }

    private static async Task<int> RunSiteDataAsync(string artifacts, string runs, string output)
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
            "site-data", "--artifacts", artifacts, "--runs", runs, "--output", output,
        })
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("No site-data process.");
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await process.WaitForExitAsync(timeout.Token);
        _ = await stdout;
        _ = await stderr;
        return process.ExitCode;
    }
}
