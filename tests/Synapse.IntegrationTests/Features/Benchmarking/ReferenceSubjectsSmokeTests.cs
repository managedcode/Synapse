using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel]
public sealed class ReferenceSubjectsSmokeTests
{
    private const string ExpectedModelSha256 =
        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
    private static readonly int[] PromptTokens = [785, 6722, 315, 9625, 374];

    [Test]
    public async Task SharedQwenFixtureMatchesPinnedDigest()
    {
        var modelPath = GetModelPath();
        await using var stream = File.OpenRead(modelPath);
        var digest = await SHA256.HashDataAsync(stream);

        await Assert.That(Convert.ToHexStringLower(digest)).IsEqualTo(ExpectedModelSha256);
    }

    [Test]
    public async Task DotLlmGeneratesTokensFromSharedQwenFixture()
    {
        var executable = RequireEnvironmentFile("SYNAPSE_DOTLLM_EXECUTABLE");
        var version = Environment.GetEnvironmentVariable("SYNAPSE_DOTLLM_VERSION")
            ?? "d88040451d7db56e5dfef9d5754ad0955b0f7fe5";
        using var result = await RunSubjectAsync(
            "dotllm",
            "--subject-executable", executable,
            "--subject-version", version);

        await AssertSuccessfulGenerationAsync(result, "dotllm");
    }

    [Test]
    public async Task LlamaSharpGeneratesTokensFromSharedQwenFixture()
    {
        using var result = await RunSubjectAsync("llamasharp", "--backend", "cpu");

        await AssertSuccessfulGenerationAsync(result, "llamasharp");
    }

    [Test]
    public async Task SynapseManagedQwen2MatchesReferenceFirstToken()
    {
        using var model = ModelLoader.Load(GetModelPath(), contextSize: 512);

        var graphVerification = ModelGraphVerifier.Verify(model.Graph);
        var result = model.Generate(PromptTokens, maximumNewTokens: 1);

        await Assert.That(graphVerification.IsValid).IsTrue();
        await Assert.That(model.Architecture).IsEqualTo("qwen2");
        await Assert.That(model.Graph.Regions.Count).IsEqualTo(26);
        await Assert.That(model.Graph.Regions.All(region =>
            region.Activation is
            {
                Decision: AlwaysActive,
                Provenance: StructuralProvenance,
                Skip: NotSkippable,
            })).IsTrue();
        await Assert.That(result.GeneratedTokens).IsEquivalentTo([12095]);
        await Assert.That(result.Elapsed).IsGreaterThan(TimeSpan.Zero);
    }

    [Test]
    public async Task PositionIsExplicitRegionInput()
    {
        using var model = ModelLoader.Load(GetModelPath(), contextSize: 512);
        var entryPoint = model.Graph.EntryPoints.Single();
        await Assert.That(entryPoint.Inputs.Count).IsEqualTo(2);
        var positionId = entryPoint.Inputs[1];
        var position = model.Graph.Values.Single(value => value.Id == positionId);
        var positionConsumers = model.Graph.Nodes.Where(node => node.Operation is
            GraphOperationKind.Rope or
            GraphOperationKind.StateAppend or
            GraphOperationKind.CausalAttention).ToArray();
        var transformerRegions = model.Graph.Regions.Where(region =>
            region.SemanticAnnotations.Contains("DenseTransformerBlock", StringComparer.Ordinal)).ToArray();

        await Assert.That(position.NumericType.Storage).IsEqualTo(StorageDataType.I32);
        await Assert.That(position.Shape.Dimensions).IsEquivalentTo([ShapeDimension.Fixed(1)]);
        await Assert.That(positionConsumers.Length).IsGreaterThan(0);
        await Assert.That(positionConsumers.All(node => node.Inputs.Contains(positionId))).IsTrue();
        await Assert.That(transformerRegions.Length).IsEqualTo(24);
        await Assert.That(transformerRegions.All(region => region.Inputs.Contains(positionId))).IsTrue();
        await Assert.That(model.Graph.Nodes
            .Where(node => node.Operation == GraphOperationKind.RmsNorm)
            .All(node => node.Attributes is NormalizationAttributes { Epsilon: > 0 })).IsTrue();
        await Assert.That(model.Graph.Nodes
            .Where(node => node.Operation == GraphOperationKind.Rope)
            .All(node => node.Attributes is RopeAttributes
            {
                Theta: > 0,
                HeadDimension: > 0,
                Layout: RotaryLayout.NeoX,
            })).IsTrue();
        await Assert.That(model.Graph.Nodes
            .Where(node => node.Operation == GraphOperationKind.CausalAttention)
            .All(node => node.Attributes is CausalAttentionAttributes
            {
                QueryHeads: > 0,
                KeyValueHeads: > 0,
                HeadDimension: > 0,
                Scale: > 0,
                Mask: AttentionMaskKind.Causal,
            })).IsTrue();
    }

    [Test]
    public async Task ContextBoundInExecutionPlanOnly()
    {
        var modelPath = GetModelPath();
        var (Fingerprint, StateContexts) = LoadGraphIdentity(modelPath, 256);
        var context512 = LoadGraphIdentity(modelPath, 512);

        await Assert.That(Fingerprint).IsEqualTo(context512.Fingerprint);
        await Assert.That(StateContexts).IsEquivalentTo(context512.StateContexts);
        await Assert.That(StateContexts.Length).IsGreaterThan(0);
        await Assert.That(StateContexts.All(dimension =>
            dimension == ShapeDimension.Bounded("Context", 1, 32768))).IsTrue();
    }

    [Test]
    public async Task RequiredWeightsResolveToSourceRanges()
    {
        var modelPath = GetModelPath();
        var modelLength = new FileInfo(modelPath).Length;
        using var model = ModelLoader.Load(modelPath, contextSize: 512);
        var descriptors = model.Graph.Weights.ToDictionary(weight => weight.Id);
        var requiredWeights = model.Graph.Regions
            .SelectMany(region => region.RequiredWeights)
            .Distinct()
            .ToArray();

        await Assert.That(descriptors.Count).IsEqualTo(requiredWeights.Length);
        await Assert.That(requiredWeights.All(descriptors.ContainsKey)).IsTrue();
        await Assert.That(descriptors.Values.All(weight =>
            weight.Source.File == "qwen2.5-0.5b-instruct-q8_0.gguf" &&
            weight.Source.Offset >= 0 &&
            weight.Source.Length > 0 &&
            weight.Source.Length <= modelLength &&
            weight.Source.Offset <= modelLength - weight.Source.Length)).IsTrue();
        await Assert.That(descriptors.Values.Select(weight => weight.Encoding).Distinct())
            .IsEquivalentTo([WeightEncoding.Fp32, WeightEncoding.GgmlQ8Zero]);
    }

    private static (ModelGraphFingerprint Fingerprint, ShapeDimension[] StateContexts) LoadGraphIdentity(
        string modelPath,
        int contextSize)
    {
        using var model = ModelLoader.Load(modelPath, contextSize);
        return (
            ModelGraphFingerprint.Compute(model.Graph),
            model.Graph.StateSlots.Select(slot => slot.Shape.Dimensions[0]).ToArray());
    }

    private static async Task AssertSuccessfulGenerationAsync(
        JsonDocument result,
        string expectedSubject)
    {
        var root = result.RootElement;
        await Assert.That(root.GetProperty("subject").GetString()).IsEqualTo(expectedSubject);
        await Assert.That(root.GetProperty("generated_tokens").GetInt32()).IsEqualTo(8);
        await Assert.That(root.GetProperty("text").GetString())
            .IsEqualTo(" Paris. It is the largest city in");
        await Assert.That(root.GetProperty("total_generation_milliseconds").GetDouble())
            .IsGreaterThan(0);
    }

}
