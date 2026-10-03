using System.Diagnostics;
using System.Text.Json.Nodes;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Website;

public sealed class SiteDataPublicationTests
{
    [Test]
    public async Task SiteDataRetainsRealNumericalEvidenceAndIndependentSourceRuns()
    {
        using var fixture = new SiteFixture();
        var raw = Path.Combine(fixture.Root, "raw", "performance-osx-arm64");
        Directory.CreateDirectory(raw);
        File.Copy(Path.Combine(FindRepositoryRoot(), "benchmarks", "results",
            "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json"),
            Path.Combine(raw, "synapse-benchmark.json"));
        var reportDirectory = Path.Combine(fixture.Root, "performance", "performance-summary");
        Directory.CreateDirectory(reportDirectory);
        await Assert.That(await RunAsync("aggregate", "--artifacts", Path.Combine(fixture.Root, "raw"),
            "--model-set", Path.Combine(FindRepositoryRoot(), "benchmarks/model-sets/foundry-local-families.json"),
            "--output", Path.Combine(fixture.Root, "summary.md"), "--json",
            Path.Combine(reportDirectory, "performance-results.json"))).IsEqualTo(3);
        using var realTests = await TestPublicationFixture.CreateAsync();
        var testsDirectory = Path.Combine(fixture.Root, "verification", $"test-results-{realTests.Runner}");
        Directory.CreateDirectory(testsDirectory);
        await Assert.That(await RunAsync("test-report", "--results", realTests.Results, "--runner", realTests.Runner,
            "--output", Path.Combine(testsDirectory, "test-results.json"))).IsEqualTo(0);
        var verificationRun = RunMetadata(36547564346, "7a219c6cb6fdb851962a3f23a2f141abde20679d");
        var performanceRun = RunMetadata(36532049579, "5b1cb73064f7b0618d963de1ab565408c30c3012");
        await fixture.WriteRunsAsync(verificationRun, performanceRun);
        await Assert.That(await fixture.PublishAsync()).IsEqualTo(0);
        var result = JsonNode.Parse(await File.ReadAllTextAsync(fixture.Output))!;
        await Assert.That(result["verification"]!["run"]!["head_sha"]!.GetValue<string>())
            .IsEqualTo(verificationRun["head_sha"]!.GetValue<string>());
        await Assert.That(result["performance"]!["run"]!["head_sha"]!.GetValue<string>())
            .IsEqualTo(performanceRun["head_sha"]!.GetValue<string>());
        var tests = result["verification"]!["reports"]!.AsArray().Single()!;
        await Assert.That(tests["runtime_identifier"]!.GetValue<string>()).IsEqualTo(realTests.Runner);
        await Assert.That(tests["total"]!.GetValue<int>()).IsEqualTo(1);
        await Assert.That(tests["passed"]!.GetValue<int>()).IsEqualTo(1);
        await Assert.That(tests["status"]!.GetValue<string>()).IsEqualTo("passed");
        var report = result["performance"]!["report"]!;
        await Assert.That(report["complete_artifacts"]!.GetValue<int>()).IsEqualTo(1);
        var rows = report["rows"]!.AsArray();
        await Assert.That(rows.Count).IsEqualTo(4);
        var native = rows.Single(row => row!["subject"]!.GetValue<string>() == "llamacpp")!;
        await Assert.That(native["decode_metric_scope"]!.GetValue<string>()).IsEqualTo("native_eval");
        await Assert.That(native["ttft_milliseconds"]).IsNull();
    }

    [Test]
    public async Task SiteDataMarksUnsupportedReportSchemaAsInvalidEvidence()
    {
        using var fixture = new SiteFixture();
        var directory = Path.Combine(fixture.Root, "verification", "test-results-osx-arm64");
        Directory.CreateDirectory(directory);
        await File.WriteAllTextAsync(Path.Combine(directory, "test-results.json"), /* language=json */ "{\"schema_version\":99}");
        await fixture.WriteRunsAsync(RunMetadata(36547564346,
            "7a219c6cb6fdb851962a3f23a2f141abde20679d"), null);
        await Assert.That(await fixture.PublishAsync()).IsEqualTo(0);
        var result = JsonNode.Parse(await File.ReadAllTextAsync(fixture.Output))!;
        await Assert.That(result["verification"]!["reports"]!.AsArray().Count).IsEqualTo(0);
        await Assert.That(result["verification"]!["missing_artifacts"]!.AsArray()[0]!.GetValue<string>())
            .Contains("invalid report");
    }

    [Test]
    public async Task SiteDataRejectsMalformedSourceRunInsteadOfTreatingItAsAbsent()
    {
        using var fixture = new SiteFixture();
        await File.WriteAllTextAsync(fixture.Runs, /* language=json */ "{\"verification\":7,\"performance\":null}");
        await Assert.That(await fixture.PublishAsync()).IsEqualTo(1);
        await Assert.That(File.Exists(fixture.Output)).IsFalse();
    }

    [Test]
    public async Task SiteDataRejectsUntrustedSourceRun()
    {
        using var fixture = new SiteFixture();
        var run = RunMetadata(36547564346, "7a219c6cb6fdb851962a3f23a2f141abde20679d");
        run["event"] = "pull_request";
        await fixture.WriteRunsAsync(run, null);
        await Assert.That(await fixture.PublishAsync()).IsEqualTo(1);
        await Assert.That(File.Exists(fixture.Output)).IsFalse();
    }

    private static JsonObject RunMetadata(long id, string sha) => new()
    {
        ["id"] = id,
        ["run_number"] = 21,
        ["run_attempt"] = 1,
        ["head_sha"] = sha,
        ["html_url"] = $"https://github.com/managedcode/Synapse/actions/runs/{id}",
        ["repository"] = "managedcode/Synapse",
        ["head_branch"] = "main",
        ["event"] = "push",
        ["conclusion"] = "failure",
        ["updated_at"] = "2026-09-29T09:35:07Z",
    };

    private static async Task<int> RunAsync(params string[] args)
    {
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"));
        foreach (var argument in args)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("No publication process.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await process.WaitForExitAsync(timeout.Token);
        _ = await output;
        _ = await error;
        return process.ExitCode;
    }

    private sealed class SiteFixture : IDisposable
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(), $"site-publication-{Guid.NewGuid():N}");
        public string Runs => Path.Combine(Root, "runs.json");
        public string Output => Path.Combine(Root, "latest.json");
        public SiteFixture()
        {
            Directory.CreateDirectory(Root);
        }

        public Task WriteRunsAsync(JsonObject? verification, JsonObject? performance) =>
            File.WriteAllTextAsync(Runs, new JsonObject
            {
                ["verification"] = verification,
                ["performance"] = performance,
            }.ToJsonString());

        public Task<int> PublishAsync() => RunAsync("site-data", "--artifacts", Root, "--runs", Runs, "--output", Output);
        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
