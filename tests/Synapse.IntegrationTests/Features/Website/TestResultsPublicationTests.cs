using System.Diagnostics;
using System.Text.Json;
using System.Xml.Linq;

namespace ManagedCode.Synapse.IntegrationTests.Features.Website;

[NotInParallel]
public sealed class TestResultsPublicationTests
{
    [Test]
    public async Task PublishesAndAggregatesRealTUnitReports()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        using var secondRun = await TestPublicationFixture.CreateAsync();
        File.Copy(secondRun.Trx, Path.Combine(fixture.Results, "second.trx"));

        var (exit, error) = await fixture.ExportAsync();
        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.Output));
        var result = json.RootElement;
        await Assert.That(result.GetProperty("schema_version").GetInt32()).IsEqualTo(1);
        await Assert.That(result.GetProperty("scope").GetString()).IsEqualTo(".NET functional tests");
        await Assert.That(result.GetProperty("status").GetString()).IsEqualTo("passed");
        await Assert.That(result.GetProperty("total").GetInt32()).IsEqualTo(2);
        await Assert.That(result.GetProperty("passed").GetInt32()).IsEqualTo(2);
        await Assert.That(result.GetProperty("failed").GetInt32()).IsEqualTo(0);
        await Assert.That(result.GetProperty("skipped").GetInt32()).IsEqualTo(0);
        await Assert.That(result.GetProperty("duration_seconds").GetDouble()).IsGreaterThanOrEqualTo(0);
        await Assert.That(result.GetProperty("runtime_identifier").GetString()).IsEqualTo(fixture.Runner);
        await Assert.That(DateTimeOffset.TryParse(result.GetProperty("generated_at_utc").GetString(), out _)).IsTrue();
        await Assert.That(result.GetProperty("issues").GetArrayLength()).IsEqualTo(0);
    }

    [Test]
    public async Task MissingReportsAndZeroTestsNeverPass()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var real = XDocument.Load(fixture.Trx);
        real.Descendants().Where(element => element.Name.LocalName == "UnitTestResult").Remove();
        foreach (var attribute in real.Descendants().Single(element => element.Name.LocalName == "Counters").Attributes())
        {
            attribute.Value = "0";
        }

        real.Save(fixture.Trx);
        await AssertRejectedAsync(fixture, "not_run", 0);
        File.Delete(fixture.Output);
        File.Delete(fixture.Trx);
        await AssertRejectedAsync(fixture, "not_run", 0);
    }

    [Test]
    public async Task FailedTestsArePublishedWithFailingExit()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var real = XDocument.Load(fixture.Trx);
        real.Descendants().Single(element => element.Name.LocalName == "UnitTestResult").SetAttributeValue("outcome", "Failed");
        var counters = real.Descendants().Single(element => element.Name.LocalName == "Counters");
        counters.SetAttributeValue("passed", 0);
        counters.SetAttributeValue("failed", 1);
        real.Save(fixture.Trx);

        await AssertRejectedAsync(fixture, "failed", 1);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.Output));
        await Assert.That(json.RootElement.GetProperty("failed").GetInt32()).IsEqualTo(1);
    }

    [Test]
    public async Task InconsistentCountersAndUnsafeXmlAreInvalid()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var real = XDocument.Load(fixture.Trx);
        real.Descendants().Single(element => element.Name.LocalName == "Counters").SetAttributeValue("passed", 2);
        real.Save(fixture.Trx);
        await AssertRejectedAsync(fixture, "invalid", 0);
        File.Delete(fixture.Output);
        var xml = await File.ReadAllTextAsync(fixture.Trx);
        var declarationEnd = xml.IndexOf("?>", StringComparison.Ordinal) + 2;
        await File.WriteAllTextAsync(fixture.Trx, xml.Insert(declarationEnd,
            "\n<!DOCTYPE TestRun [<!ENTITY unsafe SYSTEM 'file:///nonexistent'>]>"));
        await AssertRejectedAsync(fixture, "invalid", 0);
    }

    [Test]
    public async Task MissingDurationIsNullAndPublishedEvidenceIsImmutable()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var real = XDocument.Load(fixture.Trx);
        real.Descendants().Single(element => element.Name.LocalName == "Times").Remove();
        real.Save(fixture.Trx);
        var (exit, error) = await fixture.ExportAsync();
        await Assert.That(exit).IsEqualTo(0).Because(error);
        var original = await File.ReadAllBytesAsync(fixture.Output);
        using var json = JsonDocument.Parse(original);
        await Assert.That(json.RootElement.GetProperty("duration_seconds").ValueKind).IsEqualTo(JsonValueKind.Null);

        var (secondExit, _) = await fixture.ExportAsync();
        await Assert.That(secondExit).IsNotEqualTo(0);
        await Assert.That(await File.ReadAllBytesAsync(fixture.Output)).IsEquivalentTo(original);
        await Assert.That(Directory.GetFiles(fixture.Directory, "*.tmp")).IsEmpty();
    }

    [Test]
    public async Task TestRecordsMustReconcileAndDuplicateRunsAreInvalid()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var original = XDocument.Load(fixture.Trx);
        var mismatch = new XDocument(original);
        mismatch.Descendants().Single(element => element.Name.LocalName == "UnitTestResult").SetAttributeValue("outcome", "Failed");
        mismatch.Save(fixture.Trx);
        await AssertRejectedAsync(fixture, "invalid", 0);
        File.Delete(fixture.Output);
        original.Save(fixture.Trx);
        File.Copy(fixture.Trx, Path.Combine(fixture.Results, "duplicate.trx"));
        await AssertRejectedAsync(fixture, "invalid", 1);
    }

    [Test]
    public async Task SkippedOnlyAndUnsuccessfulRunOutcomesNeverPass()
    {
        using var fixture = await TestPublicationFixture.CreateAsync();
        var original = XDocument.Load(fixture.Trx);
        var skipped = new XDocument(original);
        skipped.Descendants().Single(element => element.Name.LocalName == "UnitTestResult").SetAttributeValue("outcome", "NotExecuted");
        var counters = skipped.Descendants().Single(element => element.Name.LocalName == "Counters");
        counters.SetAttributeValue("passed", 0);
        counters.SetAttributeValue("notExecuted", 1);
        counters.SetAttributeValue("executed", 0);
        skipped.Save(fixture.Trx);
        await AssertRejectedAsync(fixture, "not_run", 1);
        File.Delete(fixture.Output);
        original.Descendants().Single(element => element.Name.LocalName == "ResultSummary").SetAttributeValue("outcome", "Aborted");
        original.Save(fixture.Trx);
        await AssertRejectedAsync(fixture, "failed", 1);
    }

    private static async Task AssertRejectedAsync(TestPublicationFixture fixture, string status, int total)
    {
        var (exit, error) = await fixture.ExportAsync();
        await Assert.That(exit).IsEqualTo(3).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(fixture.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo(status);
        await Assert.That(json.RootElement.GetProperty("total").GetInt32()).IsEqualTo(total);
        await Assert.That(json.RootElement.GetProperty("issues").GetArrayLength()).IsGreaterThan(0);
    }
}

internal sealed class TestPublicationFixture : IDisposable
{
    private TestPublicationFixture()
    {
        Directory = Path.Combine(Path.GetTempPath(), $"synapse-test-publication-{Guid.NewGuid():N}");
        Results = Path.Combine(Directory, "results");
        Trx = Path.Combine(Results, "real.trx");
        Output = Path.Combine(Directory, "tests.json");
        Runner = OperatingSystem.IsMacOS() ? "osx-arm64" : OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";
        System.IO.Directory.CreateDirectory(Results);
    }

    public string Directory { get; }
    public string Results { get; }
    public string Trx { get; }
    public string Output { get; }
    public string Runner { get; }

    public static async Task<TestPublicationFixture> CreateAsync()
    {
        var fixture = new TestPublicationFixture();
        try
        {
            var (exit, error) = await RunAsync(Path.Combine(AppContext.BaseDirectory, "ManagedCode.Synapse.IntegrationTests.dll"),
                "--treenode-filter", "/*/*/GitHubPagesContractTests/PagesWorkflowPublishesCanonicalContent",
                "--report-trx", "--report-trx-filename", "real.trx", "--results-directory", fixture.Results,
                "--minimum-expected-tests", "1", "--no-ansi", "--progress", "off");
            if (exit != 0 || !File.Exists(fixture.Trx))
            {
                throw new InvalidOperationException($"Real Website fixture test exited {exit}: {error}");
            }

            return fixture;
        }
        catch
        {
            fixture.Dispose();
            throw;
        }
    }

    public Task<(int Exit, string Error)> ExportAsync() => RunAsync(
        Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"), "test-report",
        "--results", Results, "--runner", Runner, "--output", Output);

    public void Dispose() => System.IO.Directory.Delete(Directory, recursive: true);

    private static async Task<(int Exit, string Error)> RunAsync(params string[] arguments)
    {
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("Publication process could not start.");
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        var output = process.StandardOutput.ReadToEndAsync(cancellation.Token);
        var error = process.StandardError.ReadToEndAsync(cancellation.Token);
        try
        {
            await process.WaitForExitAsync(cancellation.Token);
            _ = await output;
            return (process.ExitCode, await error);
        }
        finally
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
    }
}
