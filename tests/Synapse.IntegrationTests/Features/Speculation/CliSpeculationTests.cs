using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.IntegrationTests.Features.LayerDrop;
using ManagedCode.Synapse.IntegrationTests.Features.LongContext;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.Speculation;

/// <summary><c>synapse generate --draft-drop-layers</c> (ADR-019 + ADR-020) as a real process: same tokens, reported acceptance.</summary>
[NotInParallel]
public sealed class CliSpeculationTests
{
    [Test]
    public async Task CliSpeculativeGenerateKeepsTheTargetTokens()
    {
        var model = TinyQwen2Gguf.Write(
            new TinyQwen2Gguf.Shape(Layers: 4, Heads: 2, KeyValueHeads: 1, HeadDimension: 64, FeedForward: 256, Vocabulary: 320, Context: 512),
            seed: 22);
        var compiled = Path.ChangeExtension(model, ".synapse");
        try
        {
            _ = await CompiledPackageCompiler.CompileAsync(model, compiled);
            var plain = await RunAsync("--model", compiled, "--tokens", "1,2,3,4,5,6", "--max-tokens", "24");
            var speculative = await RunAsync(
                "--model", compiled, "--tokens", "1,2,3,4,5,6", "--max-tokens", "24", "--draft-drop-layers", "2", "--draft-tokens", "4");
            var malformed = await RunAsync("--model", compiled, "--tokens", "1,2", "--draft-tokens", "4");

            await Assert.That(plain.ExitCode).IsEqualTo(0).Because(plain.Error);
            await Assert.That(speculative.ExitCode).IsEqualTo(0).Because(speculative.Error);
            using var plainJson = JsonDocument.Parse(plain.Output);
            using var speculativeJson = JsonDocument.Parse(speculative.Output);
            await Assert.That(speculativeJson.RootElement.GetProperty("generated_tokens").GetRawText())
                .IsEqualTo(plainJson.RootElement.GetProperty("generated_tokens").GetRawText());
            var speculation = speculativeJson.RootElement.GetProperty("speculation");
            await Assert.That(speculation.GetProperty("draft_profile").GetString()).EndsWith("drop1x");
            await Assert.That(speculation.GetProperty("target_passes").GetInt32()).IsGreaterThan(0).And.IsLessThanOrEqualTo(23);
            await Assert.That(malformed.ExitCode).IsEqualTo(2);
        }
        finally
        {
            File.Delete(model);
            File.Delete(compiled);
        }
    }

    private static async Task<CliRun> RunAsync(params string[] extra)
    {
        var (exitCode, output, error) = await CliScoreTests.RunCliAsync("generate", extra);
        return new CliRun(exitCode, output, error);
    }
}
