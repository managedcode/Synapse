using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

[NotInParallel]
public sealed class ContextLimitTests
{
    private const int TrainedContext = 32_768;

    [Test]
    public async Task ContextBeyondTrainedLengthFailsExplicitly()
    {
        using (var trained = Load(new ModelLoadOptions { ContextSize = TrainedContext, MaximumParallelism = 2 }))
        {
            await Assert.That(trained.ContextSize).IsEqualTo(TrainedContext);
        }

        foreach (var requested in new[] { TrainedContext + 1, 40_960, 131_072 })
        {
            var exception = await Assert.That(() => Load(new ModelLoadOptions { ContextSize = requested }))
                .Throws<NotSupportedException>();
            await Assert.That(exception!.Message).Contains(requested.ToString(System.Globalization.CultureInfo.InvariantCulture));
            await Assert.That(exception.Message).Contains("32768");
        }
    }

    [Test]
    public async Task YarnContextExtendsLimitExactly()
    {
        var yarn = RopeScaling.Yarn(4, TrainedContext);
        using var unscaled = Load(new ModelLoadOptions { ContextSize = 512, MaximumParallelism = 2 });
        using var scaled = Load(new ModelLoadOptions { ContextSize = 131_072, MaximumParallelism = 2, RopeScaling = yarn });

        await Assert.That(scaled.ContextSize).IsEqualTo(131_072);
        await Assert.That(scaled.RuntimeProfile).IsEqualTo(unscaled.RuntimeProfile + "+yarn4");
        await Assert.That(ModelGraphFingerprint.Compute(scaled.Graph))
            .IsNotEqualTo(ModelGraphFingerprint.Compute(unscaled.Graph));
        await Assert.That(scaled.Graph.Nodes.Select(node => node.Attributes).OfType<RopeAttributes>()
            .All(rope => rope.Scaling == yarn)).IsTrue();
        await Assert.That(() => Load(new ModelLoadOptions { ContextSize = 131_073, RopeScaling = yarn }))
            .Throws<NotSupportedException>();
    }

    [Test]
    public async Task ContradictingScalingFails()
    {
        var declared = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["qwen2.rope.scaling.type"] = "yarn",
            ["qwen2.rope.scaling.factor"] = 4.0f,
            ["qwen2.rope.scaling.original_context_length"] = (uint)TrainedContext,
        };

        var fromFile = RopeScalingResolver.Resolve(declared, "qwen2", requested: null);
        var matching = RopeScalingResolver.Resolve(declared, "qwen2", RopeScaling.Yarn(4, TrainedContext));
        var none = RopeScalingResolver.Resolve(new Dictionary<string, object>(), "qwen2", requested: null);

        await Assert.That(fromFile).IsEqualTo(RopeScaling.Yarn(4, TrainedContext));
        await Assert.That(matching).IsEqualTo(fromFile);
        await Assert.That(none).IsNull();
        await Assert.That(() => RopeScalingResolver.Resolve(declared, "qwen2", RopeScaling.Yarn(2, TrainedContext)))
            .Throws<NotSupportedException>();
        await Assert.That(() => RopeScalingResolver.Resolve(
                new Dictionary<string, object>(StringComparer.Ordinal) { ["qwen2.rope.scaling.type"] = "linear" },
                "qwen2",
                requested: null))
            .Throws<NotSupportedException>();
        await Assert.That(() => RopeScaling.Yarn(1, TrainedContext)).Throws<ArgumentOutOfRangeException>();
    }

    private static Qwen2Model Load(ModelLoadOptions options) =>
        (Qwen2Model)ModelLoader.Load(ReferenceBenchmarkFixture.GetModelPath(), options);
}
