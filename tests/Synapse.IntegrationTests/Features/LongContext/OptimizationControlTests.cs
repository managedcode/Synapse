using System.Text.Json;
using ManagedCode.Synapse.Cli.Features.LongContext;
using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>TEST-CTX-009-1: explicit control modes on real parsers and prepared-model child processes.</summary>
[NotInParallel]
public sealed class OptimizationControlTests
{
    [Test]
    [Arguments("off", "reference")]
    [Arguments("dense", "managed")]
    [Arguments("custom", "managed")]
    public async Task GenerateModeSelectsAndReportsItsEffectiveRuntime(string mode, string backend)
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync();
        var (exit, output, error) = await CliScoreTests.RunCliAsync("generate",
            "--model", fixture.Model, "--tokens", "1,2,3,4", "--max-tokens", "2", "--context-size", "64",
            "--threads", "1", "--optimization", mode);

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("requested_optimization").GetString()).IsEqualTo(mode);
        await Assert.That(json.RootElement.GetProperty("kernel_backend").GetString()).IsEqualTo(backend);
        await Assert.That(json.RootElement.GetProperty("runtime_profile").GetString()).Contains(backend);
        await Assert.That(json.RootElement.TryGetProperty("allow_experimental_weights", out _)).IsFalse();
        await Assert.That(json.RootElement.GetProperty("allocated_kv_bytes").GetInt64()).IsGreaterThan(0);
    }

    [Test]
    [Arguments("off", "reference")]
    [Arguments("dense", "managed")]
    public async Task ScoreModeUsesTheSameExplicitControls(string mode, string backend)
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync();
        var (exit, output, error) = await CliScoreTests.RunCliAsync("score", "--model", fixture.Model,
            "--tokens-file", fixture.Tokens, "--context-size", "64", "--first-scored", "60",
            "--threads", "1", "--optimization", mode);

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("requested_optimization").GetString()).IsEqualTo(mode);
        await Assert.That(json.RootElement.GetProperty("kernel_backend").GetString()).IsEqualTo(backend);
        await Assert.That(json.RootElement.GetProperty("kv_precision").GetString()).IsEqualTo("f32");
        await Assert.That(json.RootElement.GetProperty("scored_tokens").GetInt32()).IsEqualTo(3);
    }

    [Test]
    [Arguments("off", "--backend", "managed")]
    [Arguments("off", "--kv-precision", "f16")]
    [Arguments("off", "--kv-pages", "2:64")]
    [Arguments("off", "--drop-layers", "1")]
    [Arguments("off", "--draft-model", "draft.synapse")]
    [Arguments("dense", "--kv-precision", "f16")]
    [Arguments("dense", "--kv-pages", "2:64")]
    [Arguments("dense", "--drop-layers", "1")]
    [Arguments("dense", "--draft-drop-layers", "1")]
    [Arguments("balanced", "--threads", "1")]
    public async Task ContradictoryOrUnrecognizedModesFailBeforeModelLoading(string mode, string flag, string value)
    {
        var options = GenerationOptions.Parse(["--model", "unused.synapse", "--tokens", "1",
            "--optimization", mode, flag, value]);
        await Assert.That(options).IsNull();
    }

    [Test]
    [Arguments("--optimization", "off", "--optimization", "custom")]
    [Arguments("--optimisation", "off", "--threads", "1")]
    public async Task DuplicateOrUnknownFlagsAreNotSilentlyIgnored(string first, string value, string next, string nextValue) =>
        await Assert.That(GenerationOptions.Parse(["--model", "unused.synapse", "--tokens", "1",
            first, value, next, nextValue])).IsNull();

    [Test]
    [Arguments("--allow-experimental-weights", "true")]
    [Arguments("--weight-mean", "4")]
    public async Task RemovedWeightFlagsAreRejected(string flag, string value) =>
        await Assert.That(GenerationOptions.Parse(["--model", "unused.synapse", "--tokens", "1",
            "--optimization", "custom", flag, value])).IsNull();

    [Test]
    [Arguments("compile")]
    [Arguments("convert")]
    public async Task PreparationCommandsRejectRemovedWeightMeans(string command) =>
        await Assert.That(await ModelCommand.RunAsync([command, "--source", "unused.gguf", "--output", "unused.synapse",
            "--weight-mean", "4"])).IsEqualTo(2);

    [Test]
    public async Task ConcurrentGenerationRetainsTheRequestedModeInEvidence()
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync();
        var (exit, output, error) = await CliScoreTests.RunCliAsync("generate", "--model", fixture.Model,
            "--tokens", "1,2,3,4", "--max-tokens", "2", "--context-size", "64", "--threads", "1",
            "--optimization", "dense", "--concurrent-requests", "2");
        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("requested_optimization").GetString()).IsEqualTo("dense");
        await Assert.That(json.RootElement.GetProperty("requests").GetArrayLength()).IsEqualTo(2);
        await Assert.That(json.RootElement.GetProperty("allocated_kv_bytes").GetInt64()).IsGreaterThan(0);
    }

    [Test]
    public async Task ConcurrentGenerationRejectsSpeculationInsteadOfIgnoringTheDraft() =>
        await Assert.That(GenerationOptions.Parse(["--model", "unused.synapse", "--tokens", "1",
            "--concurrent-requests", "2", "--draft-model", "absent.synapse"])).IsNull();

    [Test]
    [Arguments("--draft-model", "absent.synapse")]
    [Arguments("--concurrent-requests", "2")]
    [Arguments("--allow-experimental-weights", "true")]
    [Arguments("--weight-mean", "4")]
    public async Task ScoreRejectsUnsupportedFlagsInCustomMode(string flag, string value)
    {
        await using var fixture = await OptimizationCliFixture.CreateAsync();
        await Assert.That(ScoreOptions.Parse([
            "--model", fixture.Model, "--tokens-file", fixture.Tokens, "--context-size", "64",
            "--optimization", "custom", flag, value])).IsNull();
    }
}

internal sealed class OptimizationCliFixture(string source, string model, string tokens) : IAsyncDisposable
{
    public string Source { get; } = source;

    public string Model { get; } = model;

    public string Tokens { get; } = tokens;

    public static async Task<OptimizationCliFixture> CreateAsync(TinyQwen2Gguf.Shape? shape = null)
    {
        var source = TinyQwen2Gguf.Write(shape ?? new(2, 2, 1, 64, 256, 320, 128), seed: 25);
        var model = Path.ChangeExtension(source, ".synapse");
        var tokens = source + ".ids";
        try
        {
            _ = await CompiledPackageCompiler.CompileAsync(source, model);
            await File.WriteAllTextAsync(tokens, string.Join(',', Enumerable.Range(0, 64).Select(index => index * 7 % 320)));
            return new(source, model, tokens);
        }
        catch
        {
            File.Delete(source);
            File.Delete(model);
            File.Delete(tokens);
            throw;
        }
    }

    public ValueTask DisposeAsync()
    {
        File.Delete(Source);
        File.Delete(Model);
        File.Delete(Tokens);
        return ValueTask.CompletedTask;
    }
}
