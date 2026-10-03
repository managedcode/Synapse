using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Actual parent/child pairing, failures and cancellation, TEST-CTX-009-3/4/5.</summary>
[NotInParallel]
public sealed class OptimizationEvaluationTests
{
    private static readonly string[] ExpectedOrder = ["dense", "reuse", "reuse", "dense"];
    [Test]
    public async Task EvaluationRotatesPairsAndPreservesIdenticalInputAndScoreTraces()
    {
        using var files = new OptimizationTestFiles();
        await WritePlanAsync(files, measurements: 1, warmups: 1);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-eval", "--plan", files.Plan, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        var samples = json.RootElement.GetProperty("samples");
        var order = samples.EnumerateArray().Select(sample => sample.GetProperty("profileId").GetString()).ToArray();
        await Assert.That(order.SequenceEqual(ExpectedOrder)).IsTrue();
        await Assert.That(samples[0].GetProperty("warmup").GetBoolean()).IsTrue();
        await Assert.That(samples[3].GetProperty("warmup").GetBoolean()).IsFalse();
        await Assert.That(json.RootElement.GetProperty("comparisons").GetArrayLength()).IsEqualTo(2);
        foreach (var pair in json.RootElement.GetProperty("comparisons").EnumerateArray())
        {
            await Assert.That(pair.GetProperty("tokenIdsIdentical").GetBoolean()).IsTrue();
            await Assert.That(pair.GetProperty("greedyAgreement").GetDouble()).IsEqualTo(1);
            await Assert.That(pair.GetProperty("meanAbsoluteNegativeLogLikelihoodDelta").GetDouble()).IsEqualTo(0);
        }

        foreach (var sample in samples.EnumerateArray())
        {
            await Assert.That(sample.GetProperty("maximumNewTokens").GetInt32()).IsEqualTo(16);
            var generations = sample.GetProperty("result").GetProperty("generations");
            var expectedReused = sample.GetProperty("profileId").GetString() == "reuse"
                ? sample.GetProperty("promptTokens").GetInt32() - 1 : 0;
            await Assert.That(generations[1].GetProperty("reusedPromptTokens").GetInt32()).IsEqualTo(expectedReused);
        }
    }

    [Test]
    public async Task FailedRealChildKeepsTheCompletedBaselineAndFailureRow()
    {
        using var files = new OptimizationTestFiles();
        await WritePlanAsync(files, measurements: 1, invalidLayer: true);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-eval", "--plan", files.Plan, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        var samples = json.RootElement.GetProperty("samples");
        await Assert.That(samples.GetArrayLength()).IsEqualTo(2);
        await Assert.That(samples[0].GetProperty("exitCode").GetInt32()).IsEqualTo(0);
        await Assert.That(samples[1].GetProperty("exitCode").GetInt32()).IsEqualTo(1);
        await Assert.That(samples[1].GetProperty("result").GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("comparisons").GetArrayLength()).IsEqualTo(0);
    }

    [Test]
    public async Task CancellationTerminatesTheRealChildAndRetainsPriorEvidence()
    {
        using var files = new OptimizationTestFiles();
        await WritePlanAsync(files, measurements: 30);
        using var cancel = new CancellationTokenSource();
        var run = OptimizationEvaluationCommand.RunAsync(["--plan", files.Plan, "--output", files.Output], cancel.Token);
        using var wait = new CancellationTokenSource(TimeSpan.FromMinutes(1));
        while (!await HasCompletedSampleAsync(files.Output))
        {
            await Task.Delay(25, wait.Token);
        }

        await cancel.CancelAsync();
        await Assert.That(await run.WaitAsync(TimeSpan.FromSeconds(15))).IsEqualTo(130);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("cancelled");
        await Assert.That(json.RootElement.GetProperty("samples").GetArrayLength()).IsGreaterThanOrEqualTo(1);
        await Assert.That(json.RootElement.GetProperty("samples")[0].GetProperty("exitCode").GetInt32()).IsEqualTo(0);
    }

    [Test]
    public async Task ChildCancellationAwaitsItsActualProcessTermination()
    {
        var path = await ReferenceBenchmarkFixture.GetCompiledModelPathAsync();
        var package = await OptimizationValidation.DescribePackageAsync(path, CancellationToken.None);
        var request = new OptimizationRequest(1, path, 4096, [.. Enumerable.Repeat(785, 2048)], ["Paris"], 16, 4, 1, false,
            new OptimizationProfile { Id = "cancel", Backend = "managed" }, package);
        using var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));

        var child = await OptimizationChildProcess.RunAsync(request, timeoutSeconds: 60, cancel.Token);

        await Assert.That(child.ExitCode).IsEqualTo(130);
        await Assert.That(() => Process.GetProcessById(child.ProcessId)).Throws<ArgumentException>();
    }

    private static async Task WritePlanAsync(OptimizationTestFiles files, int measurements, bool invalidLayer = false, int warmups = 0)
    {
        var path = await ReferenceBenchmarkFixture.GetCompiledModelPathAsync();
        var plan = OptimizationPlanTests.Plan(path);
        plan["haystackPath"] = files.Request;
        plan["contexts"] = new JsonArray(new JsonObject { ["contextSize"] = 512, ["promptTokens"] = 384 });
        plan["maximumNewTokens"] = 64;
        plan["measurements"] = measurements;
        plan["warmups"] = warmups;
        plan["threads"] = 2;
        plan["profiles"]![0]!["backend"] = "managed";
        plan["profiles"]![1]!["backend"] = "managed";
        if (invalidLayer)
        {
            plan["profiles"]![1]!["dropLayers"] = new JsonArray(200);
        }

        var root = ReferenceBenchmarkFixture.FindRepositoryRoot();
        await File.WriteAllTextAsync(files.Request, await File.ReadAllTextAsync(Path.Combine(root, "docs", "Architecture.md")) +
            "\n\n" + await File.ReadAllTextAsync(Path.Combine(root, "README.md")));
        await File.WriteAllTextAsync(files.Plan, plan.ToJsonString());
    }

    private static async Task<bool> HasCompletedSampleAsync(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(path));
        return json.RootElement.GetProperty("samples").GetArrayLength() > 0;
    }
}
