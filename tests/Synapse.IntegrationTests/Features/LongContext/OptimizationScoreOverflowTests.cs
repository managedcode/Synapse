using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>TEST-CTX-009-3 / TEST-CTX-010-4: real model scores retain finite log-space evidence at exponent limits.</summary>
[NotInParallel]
public sealed class OptimizationScoreOverflowTests
{
    private static readonly int[] ShortTokens = [0, 1, 2, 0, 1, 2];
    private static readonly int[] LongTokens = [0, 1, 2, 0, 1, 2, 0, 1];

    [Test]
    [Arguments("optimization-worker")]
    [Arguments("memory-eval")]
    public async Task FiniteScoresWithOverflowingPerplexityRetainCompleteEvidence(string command)
    {
        using var fixture = new CompiledPackageFixture(tokenizer: true);
        WriteOutputNorm(fixture.Source, 100_000f);
        _ = await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        using var files = new OptimizationTestFiles();
        await WriteRequestAsync(files, fixture.Destination, command);

        var (exit, error) = await OptimizationTestFiles.RunAsync(command, "--request", files.Request, "--output", files.Output);

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("completed");
        if (command == "optimization-worker")
        {
            await Assert.That(json.RootElement.GetProperty("generations").GetArrayLength()).IsEqualTo(2);
            await AssertOverflowScoreAsync(json.RootElement.GetProperty("score"));
        }
        else
        {
            var phases = json.RootElement.GetProperty("phases").EnumerateArray().ToArray();
            await Assert.That(phases.Last().GetProperty("name").GetString()).IsEqualTo("fresh-long-disposed");
            foreach (var phase in phases.Where(phase => phase.GetProperty("score").ValueKind == JsonValueKind.Object))
            {
                await AssertOverflowScoreAsync(phase.GetProperty("score"));
            }
            foreach (var comparison in json.RootElement.GetProperty("comparisons").EnumerateArray())
            {
                await Assert.That(comparison.GetProperty("tailPerplexityRatio").GetDouble()).IsEqualTo(1);
            }
        }
    }

    [Test]
    [Arguments("optimization-worker")]
    [Arguments("memory-eval")]
    public async Task NonfiniteScoresRetainEarlierGenerationAndMemoryEvidence(string command)
    {
        using var fixture = new CompiledPackageFixture(tokenizer: true);
        WriteOutputNorm(fixture.Source, float.MaxValue);
        _ = await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        using var files = new OptimizationTestFiles();
        await WriteRequestAsync(files, fixture.Destination, command);

        var (exit, error) = await OptimizationTestFiles.RunAsync(command, "--request", files.Request, "--output", files.Output);

        await Assert.That(exit).IsEqualTo(1).Because(error);
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("error").GetString()).Contains("nonfinite");
        if (command == "optimization-worker")
        {
            await Assert.That(json.RootElement.GetProperty("generations").GetArrayLength()).IsEqualTo(2);
            await Assert.That(json.RootElement.GetProperty("score").ValueKind).IsEqualTo(JsonValueKind.Null);
        }
        else
        {
            var phases = json.RootElement.GetProperty("phases").EnumerateArray().ToDictionary(phase => phase.GetProperty("name").GetString()!);
            await Assert.That(phases.ContainsKey("short-first")).IsTrue();
            await Assert.That(phases["disposed"].GetProperty("allocatedKvBytes").GetInt64()).IsEqualTo(0);
            await Assert.That(phases.ContainsKey("short-first-score")).IsFalse();
        }
    }

    [Test]
    [Arguments(-1000d, "underflow")]
    [Arguments(0d, "finite")]
    [Arguments(1000d, "overflow")]
    public async Task ExponentiationDescribesItsRepresentableRange(double logarithm, string expected)
    {
        var (value, status) = OptimizationScoreFactory.Exponentiate(logarithm);
        await Assert.That(status).IsEqualTo(expected);
        if (expected == "finite")
        {
            await Assert.That(value).IsEqualTo(1d);
        }
        else
        {
            await Assert.That(value).IsNull();
        }
    }

    private static async Task AssertOverflowScoreAsync(JsonElement score)
    {
        var mean = score.GetProperty("meanNegativeLogLikelihood").GetDouble();
        await Assert.That(double.IsFinite(mean)).IsTrue();
        await Assert.That(mean).IsGreaterThan(Math.Log(double.MaxValue));
        await Assert.That(score.GetProperty("logPerplexity").GetDouble()).IsEqualTo(mean);
        await Assert.That(score.GetProperty("tailPerplexity").ValueKind).IsEqualTo(JsonValueKind.Null);
        await Assert.That(score.GetProperty("perplexityStatus").GetString()).IsEqualTo("overflow");
        await Assert.That(score.GetProperty("negativeLogLikelihoods").EnumerateArray().All(value => double.IsFinite(value.GetDouble()))).IsTrue();
    }

    private static async Task WriteRequestAsync(OptimizationTestFiles files, string path, string command)
    {
        if (command == "optimization-worker")
        {
            await files.WriteRequestAsync(path, 32, ShortTokens);
            return;
        }

        await File.WriteAllTextAsync(files.Request, JsonSerializer.Serialize(new
        {
            schemaVersion = 1,
            modelPath = path,
            contextSize = 32,
            shortPromptTokens = ShortTokens,
            longPromptTokens = LongTokens,
            maximumNewTokens = 2,
            scoredTailTokens = 4,
            threads = 1,
            profile = new { id = "overflow", backend = "reference", kvPrecision = "f32", reusePromptPrefix = false },
        }));
    }

    private static void WriteOutputNorm(string path, float norm)
    {
        var tensor = GgufHeaderReader.Read(path).Tensors["output_norm.weight"];
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        stream.Position = tensor.Offset;
        for (long index = 0; index < tensor.ByteLength / sizeof(float); index++)
        {
            writer.Write(norm);
        }
    }
}
