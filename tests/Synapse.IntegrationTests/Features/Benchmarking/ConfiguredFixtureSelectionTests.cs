using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

// TEST-BMK-001-5: pipeline environment selection reaches real mapped model sources and the Foundry plan.
[NotInParallel]
public sealed class ConfiguredFixtureSelectionTests
{
    [Test]
    public async Task ConfiguredGgufSelectionReadsEachSelectedRealSource()
    {
        using var settings = new FixtureEnvironment("SYNAPSE_MODEL_ROOT", "SYNAPSE_GGUF_MODEL_ID", "SYNAPSE_GGUF_MODEL_FILE");
        var directory = Directory.CreateTempSubdirectory("synapse-selected-model-");
        var identities = new List<string>();
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_MODEL_ROOT", directory.FullName);
            for (var seed = 9201; seed <= 9202; seed++)
            {
                var modelId = $"selected-model-{seed}";
                var modelFile = $"selected-weights-{seed}.gguf";
                var destination = Path.Combine(directory.FullName, modelId, modelFile);
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                File.Move(TinyQwen2Gguf.Write(new(1, 1, 1, 64, 256, 64, 32), seed), destination);
                Environment.SetEnvironmentVariable("SYNAPSE_GGUF_MODEL_ID", modelId);
                Environment.SetEnvironmentVariable("SYNAPSE_GGUF_MODEL_FILE", modelFile);
                await Assert.That(ReferenceBenchmarkFixture.GetModelPath()).IsEqualTo(destination);
                using var actual = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
                await Assert.That(actual.SourceFile).IsEqualTo(modelFile);
                await Assert.That(actual.Tensors.Count).IsGreaterThan(0);
                await Assert.That(actual.Metadata["general.architecture"]).IsEqualTo("qwen2");
                identities.Add(Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(destination))));
            }

            await Assert.That(identities.Distinct(StringComparer.Ordinal).Count()).IsEqualTo(2)
                .Because("each selected path maps its own real source bytes");
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    [Test]
    public async Task ConfiguredFoundrySetAndAnchorReachTheRealPlan()
    {
        using var settings = new FixtureEnvironment("SYNAPSE_FOUNDRY_MODEL_SET", "SYNAPSE_FOUNDRY_ANCHOR");
        var set = ReferenceBenchmarkFixture.BenchmarkInputPath("ModelSets", "foundry-local-families.json");
        using var definition = JsonDocument.Parse(await File.ReadAllTextAsync(set));
        var selected = definition.RootElement.GetProperty("models").EnumerateArray().Last().GetProperty("alias").GetString()
            ?? throw new InvalidDataException("Configured model set has no selected alias.");
        var configuredSet = Path.Combine(Path.GetTempPath(), $"selected-foundry-set-{Guid.NewGuid():N}.json");
        File.Copy(set, configuredSet);
        try
        {
            Environment.SetEnvironmentVariable("SYNAPSE_FOUNDRY_MODEL_SET", configuredSet);
            Environment.SetEnvironmentVariable("SYNAPSE_FOUNDRY_ANCHOR", selected);
            await Assert.That(FoundryLocalFixture.ModelSetPath).IsEqualTo(configuredSet);
            await Assert.That(FoundryLocalFixture.AnchorAlias).IsEqualTo(selected);
            var result = await FoundryLocalFixture.RunAsync("plan", "--set", FoundryLocalFixture.ModelSetPath);
            await Assert.That(result.ExitCode).IsEqualTo(0).Because(result.StandardError);
            using var matrix = JsonDocument.Parse(result.StandardOutput);
            await Assert.That(matrix.RootElement.GetProperty("include").EnumerateArray()
                .Any(entry => entry.GetProperty("alias").GetString() == selected)).IsTrue();
        }
        finally
        {
            File.Delete(configuredSet);
        }
    }
}

internal sealed class FixtureEnvironment(params string[] names) : IDisposable
{
    private readonly Dictionary<string, string?> _original = names.ToDictionary(
        name => name, Environment.GetEnvironmentVariable, StringComparer.Ordinal);

    public void Dispose()
    {
        foreach (var (name, value) in _original)
        {
            Environment.SetEnvironmentVariable(name, value);
        }
    }
}
