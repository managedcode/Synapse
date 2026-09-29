using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.IntegrationTests.Features.LongContext;

namespace ManagedCode.Synapse.IntegrationTests.Features.LayerDrop;

/// <summary><c>--drop-layers</c> (experimental) and <c>--drop-profile</c> (qualified by an evidence file) as real processes.</summary>
[NotInParallel]
public sealed class CliLayerDropTests
{
    [Test]
    public async Task CliGenerateAndScoreAcceptLayerDrops()
    {
        var model = TinyQwen2Gguf.Write(
            new TinyQwen2Gguf.Shape(Layers: 4, Heads: 2, KeyValueHeads: 1, HeadDimension: 64, FeedForward: 256, Vocabulary: 320, Context: 512),
            seed: 21);
        var evidence = Path.Combine(Path.GetTempPath(), $"synapse-drop-{Guid.NewGuid():N}.json");
        var tokens = Path.Combine(Path.GetTempPath(), $"synapse-drop-{Guid.NewGuid():N}.ids");
        await File.WriteAllTextAsync(evidence, /*lang=json,strict*/ """{ "qualifiedDrop": { "layers": [2] } }""");
        await File.WriteAllTextAsync(tokens, string.Join('\n', Enumerable.Range(0, 80).Select(index => index * 7 % 320)));
        try
        {
            var experimental = await RunAsync(
                "generate", "--model", model, "--tokens", "1,2,3,4", "--max-tokens", "3", "--drop-layers", "1,2");
            var qualified = await RunAsync(
                "generate", "--model", model, "--tokens", "1,2,3,4", "--max-tokens", "3", "--drop-profile", evidence);
            var scored = await RunAsync(
                "score", "--model", model, "--tokens-file", tokens, "--context-size", "64", "--backend", "managed",
                "--drop-layers", "3");
            var invalid = await RunAsync(
                "generate", "--model", model, "--tokens", "1,2", "--drop-layers", "0,1,2,3");

            await Assert.That(experimental.ExitCode).IsEqualTo(0).Because(experimental.Error);
            await Assert.That(Subject(experimental.Output)).EndsWith("+drop2x");
            await Assert.That(qualified.ExitCode).IsEqualTo(0).Because(qualified.Error);
            await Assert.That(Subject(qualified.Output)).EndsWith("+drop1");
            await Assert.That(qualified.Error).Contains(Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(evidence))));
            await Assert.That(scored.ExitCode).IsEqualTo(0).Because(scored.Error);
            await Assert.That(Subject(scored.Output)).EndsWith("+drop1x");
            await Assert.That(invalid.ExitCode).IsNotEqualTo(0);
            await Assert.That(invalid.Error).Contains("At least one layer must stay");
        }
        finally
        {
            File.Delete(model);
            File.Delete(evidence);
            File.Delete(tokens);
        }
    }

    private static async Task<CliRun> RunAsync(string command, params string[] extra)
    {
        var (exitCode, output, error) = await CliScoreTests.RunCliAsync(command, extra);
        return new CliRun(exitCode, output, error);
    }

    private static string Subject(string output)
    {
        using var json = JsonDocument.Parse(output);
        return json.RootElement.GetProperty("subject").GetString()!;
    }
}

internal sealed record CliRun(int ExitCode, string Output, string Error);
