using System.Text.Json;
using System.Text.Json.Nodes;
using ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Strict prepared-only plan contracts, TEST-CTX-009-2/5.</summary>
[NotInParallel]
public sealed class OptimizationPlanTests
{
    [Test]
    [Arguments("unknown")]
    [Arguments("duplicateBaseline")]
    [Arguments("cpuFp16")]
    [Arguments("source")]
    [Arguments("crossBackend")]
    public async Task InvalidPlanRetainsExplicitFailureEvidence(string invalid)
    {
        using var model = new CompiledPackageFixture(tokenizer: true);
        _ = await CompiledPackageCompiler.CompileAsync(model.Source, model.Destination);
        using var files = new OptimizationTestFiles();
        var plan = Plan(model.Destination);
        switch (invalid)
        {
            case "unknown": plan["automaticSpeedup"] = true; break;
            case "duplicateBaseline": plan["profiles"]![1]!["id"] = "dense"; break;
            case "cpuFp16": plan["profiles"]![1]!["kvPrecision"] = "f16"; break;
            case "source": plan["modelPath"] = model.Source; break;
            case "crossBackend": plan["profiles"]![1]!["backend"] = "managed"; break;
            default: throw new ArgumentOutOfRangeException(nameof(invalid));
        }

        await File.WriteAllTextAsync(files.Plan, plan.ToJsonString());
        var (exitCode, error) = await OptimizationTestFiles.RunAsync("optimization-eval", "--plan", files.Plan, "--output", files.Output);

        await Assert.That(exitCode).IsEqualTo(2).Because(error);
        await Assert.That(File.Exists(files.Output)).IsTrue();
        using var json = JsonDocument.Parse(await File.ReadAllTextAsync(files.Output));
        await Assert.That(json.RootElement.GetProperty("status").GetString()).IsEqualTo("failed");
        await Assert.That(json.RootElement.GetProperty("samples").GetArrayLength()).IsEqualTo(0);
        var reason = invalid switch
        {
            "unknown" => "automaticSpeedup",
            "duplicateBaseline" => "unique",
            "cpuFp16" => "FP16 KV on CPU",
            "source" => ".synapse",
            "crossBackend" => "same backend",
            _ => throw new ArgumentOutOfRangeException(nameof(invalid)),
        };
        await Assert.That(json.RootElement.GetProperty("error").GetString()).Contains(reason);
    }

    [Test]
    public async Task SourceGeneratedPlanPreservesAbsentOptionalDefaults()
    {
        using var files = new OptimizationTestFiles();
        var input = Plan("/temporary/prepared.synapse");
        _ = input["profiles"]![0]!.AsObject().Remove("backend");
        _ = input["profiles"]![1]!.AsObject().Remove("backend");
        _ = input["profiles"]![0]!.AsObject().Remove("kvPrecision");
        await File.WriteAllTextAsync(files.Plan, input.ToJsonString());

        var plan = await OptimizationValidation.ReadPlanAsync(files.Plan, CancellationToken.None);

        await Assert.That(plan.TimeoutSeconds).IsEqualTo(900);
        await Assert.That(plan.Seed).IsEqualTo(20261003);
        await Assert.That(plan.IncludeRepeatedPrompt).IsTrue();
        await Assert.That(plan.Profiles[0].KvPrecision).IsEqualTo("f32");
        await Assert.That(plan.Profiles[0].Backend).IsEqualTo("managed");
        await Assert.That(plan.Profiles[1].Backend).IsEqualTo("managed");
    }

    [Test]
    public async Task SourceGeneratedPlanPreservesKvPageDefaults()
    {
        using var files = new OptimizationTestFiles();
        var input = Plan("/temporary/prepared.synapse");
        input["profiles"]![0]!["backend"] = "managed";
        input["profiles"]![1]!["backend"] = "managed";
        input["profiles"]![1]!["kvPages"] = new JsonObject { ["budgetPages"] = 1, ["windowTokens"] = 16 };
        await File.WriteAllTextAsync(files.Plan, input.ToJsonString());

        var plan = await OptimizationValidation.ReadPlanAsync(files.Plan, CancellationToken.None);
        var pages = plan.Profiles[1].KvPages!.ToRuntime();

        await Assert.That(pages.PageTokens).IsEqualTo(64);
        await Assert.That(pages.Selection).IsEqualTo(KvPageSelection.KeyBound);
        await Assert.That(pages.Seed).IsEqualTo(0);
        await Assert.That(pages.BudgetPages).IsEqualTo(1);
        await Assert.That(pages.WindowTokens).IsEqualTo(16);
    }

    internal static JsonObject Plan(string modelPath) => new()
    {
        ["schemaVersion"] = 1,
        ["modelPath"] = modelPath,
        ["haystackPath"] = modelPath,
        ["baselineProfile"] = "dense",
        ["contexts"] = new JsonArray(new JsonObject { ["contextSize"] = 32, ["promptTokens"] = 16 }),
        ["tasks"] = new JsonArray("needle"),
        ["depths"] = new JsonArray(0.5),
        ["maximumNewTokens"] = 2,
        ["scoredTailTokens"] = 4,
        ["warmups"] = 0,
        ["measurements"] = 1,
        ["threads"] = 1,
        ["profiles"] = new JsonArray(
            new JsonObject { ["id"] = "dense", ["backend"] = "reference", ["kvPrecision"] = "f32" },
            new JsonObject { ["id"] = "reuse", ["backend"] = "reference", ["kvPrecision"] = "f32", ["reusePromptPrefix"] = true }),
    };
}
