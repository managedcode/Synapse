using System.Text.Json;
using System.Text.Json.Nodes;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Real prepared-model memory phases, quality, cancellation and publication, TEST-CTX-010-4.</summary>
[NotInParallel]
public sealed class MemoryEvaluationTests
{
    [Test]
    [Arguments(false, 2)]
    [Arguments(true, 0)]
    public async Task MemoryPhasesContractAndMatchFreshModelMath(bool reuse, int scoredTail)
    {
        using var model = new MemoryTestModel();
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        await WriteRequestAsync(files, model.Destination, reuse, scoredTail);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("memory-eval", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        var evidence = json.RootElement;
        await Assert.That(evidence.GetProperty("status").GetString()).IsEqualTo("completed");
        await Assert.That(evidence.GetProperty("package").GetProperty("sourceSha256").GetString()).IsEqualTo(
            CompiledPackageReader.Inspect(model.Destination).SourceSha256);
        var phases = evidence.GetProperty("phases").EnumerateArray().ToDictionary(phase => phase.GetProperty("name").GetString()!);
        var shortBytes = phases["short-first"].GetProperty("allocatedKvBytes").GetInt64();
        await Assert.That(phases["loaded"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
        await Assert.That(shortBytes).IsGreaterThan(0);
        await Assert.That(phases["long"].GetProperty("allocatedKvBytes").GetInt64()).IsGreaterThanOrEqualTo(shortBytes * 4);
        await Assert.That(phases["short-after-long"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(shortBytes);
        await Assert.That(phases["disposed"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
        await Assert.That(phases["fresh-short-disposed"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
        await Assert.That(phases["fresh-long-disposed"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
        await Assert.That(phases["long"].GetProperty("workingSetBytes").GetInt64()).IsGreaterThan(0);
        await Assert.That(evidence.GetProperty("comparisons").GetArrayLength()).IsEqualTo(3);
        foreach (var comparison in evidence.GetProperty("comparisons").EnumerateArray())
        {
            await Assert.That(comparison.GetProperty("promptTokenIdsIdentical").GetBoolean()).IsTrue();
            await Assert.That(comparison.GetProperty("generatedTokensIdentical").GetBoolean()).IsTrue();
            if (scoredTail > 0)
            {
                await Assert.That(comparison.GetProperty("meanAbsoluteNegativeLogLikelihoodDelta").GetDouble()).IsEqualTo(0);
                await Assert.That(comparison.GetProperty("greedyAgreement").GetDouble()).IsEqualTo(1);
            }
        }
    }

    [Test]
    [Arguments("source")]
    [Arguments("bounds")]
    [Arguments("unknown")]
    [Arguments("profilePath")]
    public async Task InvalidMemoryRequestRetainsFailureWithoutExecution(string kind)
    {
        using var model = new MemoryTestModel();
        using var files = new OptimizationTestFiles();
        await WriteRequestAsync(files, kind == "source" ? model.Source : model.Destination, reuse: false, scoredTail: 0);
        var request = JsonNode.Parse(await File.ReadAllTextAsync(files.Request))!;
        if (kind == "bounds")
        {
            request["contextSize"] = 4;
        }
        else if (kind == "unknown")
        {
            request["forceOptimal"] = true;
        }
        else if (kind == "profilePath")
        {
            request["profile"]!["modelPath"] = "/forged.synapse";
        }
        await File.WriteAllTextAsync(files.Request, request.ToJsonString());

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("memory-eval", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("phases").GetArrayLength()).IsEqualTo(0);
        var expected = kind switch
        {
            "source" => ".synapse",
            "bounds" => "bounded",
            "profilePath" => "profile modelPath",
            _ => "forceOptimal",
        };
        await Assert.That(json.RootElement.GetProperty("error").GetString()).Contains(expected);
        await Assert.That(File.Exists(model.Destination)).IsFalse();
    }

    [Test]
    public async Task MemoryEvaluationPreservesPublishedSentinel()
    {
        using var files = new OptimizationTestFiles();
        await File.WriteAllTextAsync(files.Output, "sentinel evidence");

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("memory-eval", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        await Assert.That(await File.ReadAllTextAsync(files.Output)).IsEqualTo("sentinel evidence");
    }

    [Test]
    public async Task MemoryCancellationKeepsCompletedShortPhaseAndDisposesKv()
    {
        using var model = new MemoryTestModel(context: 16384);
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        await WriteRequestAsync(files, model.Destination, reuse: true, scoredTail: 0, longTokens: 16370, context: 16384);
        using var cancel = new CancellationTokenSource();
        var run = MemoryEvaluationCommand.RunAsync(["--request", files.Request, "--output", files.Output], cancel.Token);
        using var wait = new CancellationTokenSource(TimeSpan.FromMinutes(1));
        while (!await HasShortPhaseAsync(files.Output))
        {
            await Task.Delay(25, wait.Token);
        }

        await cancel.CancelAsync();
        await Assert.That(await run.WaitAsync(TimeSpan.FromSeconds(15))).IsEqualTo(130);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("cancelled");
        var phases = json.RootElement.GetProperty("phases").EnumerateArray().ToDictionary(phase => phase.GetProperty("name").GetString()!);
        await Assert.That(phases.ContainsKey("short-first")).IsTrue();
        await Assert.That(phases["disposed"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
    }

    private static async Task WriteRequestAsync(OptimizationTestFiles files, string modelPath, bool reuse, int scoredTail,
        int longTokens = 3073, int context = 4096) =>
        await File.WriteAllTextAsync(files.Request, JsonSerializer.Serialize(new
        {
            schemaVersion = 1,
            modelPath,
            contextSize = context,
            shortPromptTokens = Enumerable.Range(0, 6).Select(index => index % 3).ToArray(),
            longPromptTokens = Enumerable.Range(0, longTokens).Select(index => index % 3).ToArray(),
            maximumNewTokens = 2,
            scoredTailTokens = scoredTail,
            threads = 1,
            profile = new { id = "memory", backend = "managed", kvPrecision = "f32", reusePromptPrefix = reuse },
        }));

    private static async Task<bool> HasShortPhaseAsync(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(path));
        return json.RootElement.GetProperty("phases").EnumerateArray().Any(phase => phase.GetProperty("name").GetString() == "short-first");
    }
}

internal sealed class MemoryTestModel(int context = 4096) : IDisposable
{
    internal string Source { get; } = TinyQwen2Gguf.Write(new(1, 1, 1, 32, 32, 32, context), seed: 47);
    internal string Destination => Path.ChangeExtension(Source, ".synapse");

    public void Dispose()
    {
        File.Delete(Source);
        File.Delete(Destination);
    }
}
