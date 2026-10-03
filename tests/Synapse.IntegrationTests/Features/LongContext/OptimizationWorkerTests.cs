using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Real child-process generation and scoring, TEST-CTX-009-3/4/5.</summary>
[NotInParallel]
public sealed class OptimizationWorkerTests
{
    [Test]
    public async Task WorkerRecordsAlignedScoresAndControlledRepeatedRequest()
    {
        using var model = new CompiledPackageFixture(tokenizer: true);
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        await files.WriteRequestAsync(model.Destination, context: 32, tokens: [0, 1, 2, 0, 1, 2]);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-worker", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        var result = json.RootElement;
        await Assert.That(result.GetProperty("status").GetString()).IsEqualTo("completed");
        await Assert.That(result.GetProperty("sourceSha256").GetString()).IsEqualTo(
            CompiledPackageReader.Inspect(model.Destination).SourceSha256);
        await Assert.That(result.GetProperty("promptTokenIdsSha256").GetString()!.Length).IsEqualTo(64);
        var cold = result.GetProperty("generations")[0];
        var repeat = result.GetProperty("generations")[1];
        await Assert.That(cold.GetProperty("prefillMilliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(cold.GetProperty("reusedPromptTokens").GetInt32()).IsEqualTo(0);
        await Assert.That(repeat.GetProperty("reusedPromptTokens").GetInt32()).IsEqualTo(5);
        await Assert.That(repeat.GetProperty("generatedTokens").GetRawText()).IsEqualTo(cold.GetProperty("generatedTokens").GetRawText());
        var score = result.GetProperty("score");
        await Assert.That(score.GetProperty("firstScoredPosition").GetInt32()).IsEqualTo(1);
        await Assert.That(score.GetProperty("negativeLogLikelihoods").GetArrayLength()).IsEqualTo(4);
        await Assert.That(score.GetProperty("greedyTokens").GetArrayLength()).IsEqualTo(4);
        await Assert.That(score.GetProperty("tailPerplexity").GetDouble()).IsGreaterThan(0);
        await Assert.That(result.GetProperty("graphFingerprint").GetString()!.Length).IsEqualTo(64);
    }

    [Test]
    public async Task WorkerRetainsFailureAndNeverConvertsSourceModel()
    {
        using var model = new CompiledPackageFixture(tokenizer: true);
        using var files = new OptimizationTestFiles();
        await files.WriteRequestAsync(model.Source, context: 32, tokens: [0, 1, 2]);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-worker", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("error").GetString()).Contains(".synapse");
        await Assert.That(File.Exists(model.Destination)).IsFalse();
    }

    [Test]
    public async Task WorkerRejectsPromptAndOutputBeyondContextWithoutSuccessfulRow()
    {
        using var model = new CompiledPackageFixture(tokenizer: true);
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        await files.WriteRequestAsync(model.Destination, context: 4, tokens: [0, 1, 2]);

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-worker", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("generations").GetArrayLength()).IsEqualTo(0);
    }

    [Test]
    public async Task WorkerPreservesAlreadyPublishedEvidence()
    {
        using var files = new OptimizationTestFiles();
        await File.WriteAllTextAsync(files.Output, "sentinel evidence");

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-worker", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        await Assert.That(await File.ReadAllTextAsync(files.Output)).IsEqualTo("sentinel evidence");
    }

    [Test]
    [Arguments("identity")]
    [Arguments("sourceSha256")]
    [Arguments("fileSha256")]
    [Arguments("graphFingerprint")]
    public async Task WorkerRejectsForgedPackageProvenance(string field)
    {
        using var model = new CompiledPackageFixture(tokenizer: true);
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        await files.WriteRequestAsync(model.Destination, context: 32, tokens: [0, 1, 2]);
        var request = JsonNode.Parse(await File.ReadAllTextAsync(files.Request))!;
        request["package"]![field] = JsonValue.Create(new string('0', 64));
        await File.WriteAllTextAsync(files.Request, request.ToJsonString());

        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-worker", "--request", files.Request, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("error").GetString()).Contains("provenance");
        await Assert.That(json.RootElement.GetProperty("generations").GetArrayLength()).IsEqualTo(0);
    }
}

internal sealed class OptimizationTestFiles : IDisposable
{
    private static readonly string[] Answers = ["ab"];
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "synapse-optimization-test-" + Guid.NewGuid().ToString("N"));

    internal OptimizationTestFiles()
    {
        _ = Directory.CreateDirectory(_directory);
    }
    internal string Request => Path.Combine(_directory, "request.json");
    internal string Output => Path.Combine(_directory, "output.json");
    internal string Plan => Path.Combine(_directory, "plan.json");

    internal async Task WriteRequestAsync(string modelPath, int context, int[] tokens)
    {
        var package = modelPath.EndsWith(".synapse", StringComparison.Ordinal)
            ? await OptimizationValidation.DescribePackageAsync(modelPath, CancellationToken.None) : null;
        await File.WriteAllTextAsync(Request, JsonSerializer.Serialize(new
        {
            schemaVersion = 1,
            modelPath,
            contextSize = context,
            promptTokens = tokens,
            answers = Answers,
            maximumNewTokens = 2,
            scoredTailTokens = 4,
            threads = 1,
            includeRepeatedPrompt = true,
            profile = new { id = "reuse", backend = "reference", kvPrecision = "f32", reusePromptPrefix = true },
            package,
        }, JsonSerializerOptions.Web));
    }

    internal static async Task<(int ExitCode, string Error)> RunAsync(params string[] arguments)
    {
        var start = new ProcessStartInfo("dotnet") { RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false };
        start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"));
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("Benchmark child did not start.");
        var error = process.StandardError.ReadToEndAsync();
        var output = process.StandardOutput.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));
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

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
