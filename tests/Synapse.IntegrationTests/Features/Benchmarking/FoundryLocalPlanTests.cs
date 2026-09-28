using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.FoundryLocalFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class FoundryLocalPlanTests
{
    private static readonly string[] RequiredFamilies = ["qwen", "phi", "mistral", "deepseek"];

    [Test]
    public async Task FoundryPlanSchedulesOneIsolatedJobPerRunnerAndModel()
    {
        var summary = Path.Combine(Path.GetTempPath(), $"foundry-plan-{Guid.NewGuid():N}.md");
        try
        {
            var result = await RunAsync("plan", "--set", ModelSetPath, "--summary", summary);

            await Assert.That(result.ExitCode).IsEqualTo(0).Because(result.StandardError);
            await Assert.That(result.StandardOutput.Trim()).DoesNotContain('\n');
            using var matrix = JsonDocument.Parse(result.StandardOutput);
            using var set = JsonDocument.Parse(await File.ReadAllTextAsync(ModelSetPath));
            var expected = FittingPairs(set.RootElement);
            var scheduled = matrix.RootElement.GetProperty("include").EnumerateArray()
                .Select(entry => (Runner: entry.GetProperty("runner").GetString()!,
                    Alias: entry.GetProperty("alias").GetString()!,
                    Family: entry.GetProperty("family").GetString()!))
                .ToArray();

            await Assert.That(scheduled.Select(entry => (entry.Runner, entry.Alias)).ToHashSet().SetEquals(expected))
                .IsTrue().Because("each runner receives exactly the models that fit half of its memory");
            await Assert.That(scheduled.Length).IsEqualTo(expected.Count)
                .Because("every matrix entry is one isolated runner-and-model job");
            foreach (var family in RequiredFamilies)
            {
                await Assert.That(scheduled.Any(entry => entry.Family == family)).IsTrue()
                    .Because($"the '{family}' family must be measured on at least one runner");
            }

            var coverage = await File.ReadAllTextAsync(summary);
            await Assert.That(coverage).Contains("|macos-15|deepseek-r1-7b|excluded: 6584 MB > half of 7168 MB|");
            await Assert.That(coverage).Contains("|ubuntu-24.04|deepseek-r1-7b|scheduled|");
        }
        finally
        {
            File.Delete(summary);
        }
    }

    [Test]
    [Arguments("duplicate", "duplicate alias 'qwen2.5-0.5b'")]
    [Arguments("oversized", "fits no runner")]
    [Arguments("no-cpu", "has no cpu variant")]
    [Arguments("bad-system", "unknown systemPrompt 'merge'")]
    public async Task FoundryPlanRejectsInvalidModelSets(string defect, string message)
    {
        var path = Path.Combine(Path.GetTempPath(), $"foundry-set-{defect}-{Guid.NewGuid():N}.json");
        try
        {
            await File.WriteAllTextAsync(path, InvalidSet(defect));

            var result = await RunAsync("plan", "--set", path);

            await Assert.That(result.ExitCode).IsEqualTo(1);
            await Assert.That(result.StandardError).Contains(message);
            await Assert.That(result.StandardOutput).IsEmpty();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task WorkflowsIsolateFoundryLocalJobs()
    {
        var workflows = Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), ".github", "workflows");
        var performance = await File.ReadAllTextAsync(Path.Combine(workflows, "performance.yml"));
        var verify = await File.ReadAllTextAsync(Path.Combine(workflows, "verify.yml"));

        await Assert.That(performance).Contains("push:\n    branches: [main]");
        await Assert.That(performance).Contains("workflow_dispatch:");
        await Assert.That(performance).Contains("foundry-local-plan:");
        await Assert.That(performance).Contains("matrix: ${{ fromJSON(needs.foundry-local-plan.outputs.matrix) }}");
        await Assert.That(performance).Contains("runs-on: ${{ matrix.runner }}");
        await Assert.That(performance).Contains("fetch --set benchmarks/model-sets/foundry-local-families.json --alias \"${{ matrix.alias }}\"");
        await Assert.That(performance).Contains("name: foundry-local-${{ matrix.runner }}-${{ matrix.alias }}");
        await Assert.That(performance).Contains("dotnet restore experiments/Synapse.FoundryLocalBenchmarks --locked-mode");
        await Assert.That(performance).Contains("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1");
        await Assert.That(performance).DoesNotContain("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02");
        await Assert.That(verify).Contains("--alias qwen2.5-0.5b");
        await Assert.That(verify).Contains("SYNAPSE_FOUNDRY_CACHE:");
        await Assert.That(verify).DoesNotContain("--alias deepseek-r1-7b");
    }

    private static HashSet<(string Runner, string Alias)> FittingPairs(JsonElement set)
    {
        var pairs = new HashSet<(string, string)>();
        foreach (var runner in set.GetProperty("runners").EnumerateArray())
        {
            var memory = runner.GetProperty("memoryMb").GetInt32();
            foreach (var model in set.GetProperty("models").EnumerateArray())
            {
                var cpu = model.GetProperty("variants").EnumerateArray()
                    .Single(variant => variant.GetProperty("device").GetString() == "cpu");
                if (cpu.GetProperty("fileSizeMb").GetInt32() * 2 <= memory)
                {
                    _ = pairs.Add((runner.GetProperty("id").GetString()!, model.GetProperty("alias").GetString()!));
                }
            }
        }

        return pairs;
    }

    private static string InvalidSet(string defect)
    {
        var second = defect switch
        {
            "duplicate" => Model("qwen2.5-0.5b", "cpu", 822),
            "oversized" => Model("huge", "cpu", 999_999),
            "bad-system" => Model("templated", "cpu", 822, "\"systemPrompt\": \"merge\", "),
            _ => Model("gpu-only", "gpu", 700),
        };
        return $$"""
            {
              "schemaVersion": 1,
              "id": "invalid",
              "contextTokens": 1024,
              "memoryRule": "half",
              "runners": [ { "id": "ubuntu-24.04", "name": "Ubuntu", "memoryMb": 16384 } ],
              "models": [ {{Model("qwen2.5-0.5b", "cpu", 822)}}, {{second}} ]
            }
            """;
    }

    private static string Model(string alias, string device, int size, string extra = "") => $$"""
        { "family": "qwen", "alias": "{{alias}}", "note": "test", {{extra}}
          "variants": [ { "device": "{{device}}", "id": "{{alias}}-generic-{{device}}:1", "fileSizeMb": {{size}} } ] }
        """;
}
