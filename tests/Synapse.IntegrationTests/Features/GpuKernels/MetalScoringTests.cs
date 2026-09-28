using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.IntegrationTests.Features.LongContext;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>Metal teacher-forced scoring against the FP32 reference oracle (ADR-015).</summary>
[NotInParallel]
public sealed class MetalScoringTests
{
    [Test]
    [Arguments(3)]
    [Arguments(64)]
    public async Task MetalScoreTracksReference(int scoringRows)
    {
        GpuHardware.RequireMetal();
        var tokens = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath()).Encode(TokenScoringTests.Text);
        using var reference = TokenScoringTests.Load(KernelBackend.Reference, scoringRows: 8);
        using var metal = TokenScoringTests.Load(KernelBackend.Metal, scoringRows);

        var expected = reference.Score(tokens, 2, progress: null);
        var actual = metal.Score(tokens, 2, progress: null);

        await Assert.That(actual.GreedyTokens).IsEquivalentTo(expected.GreedyTokens);
        for (var index = 0; index < expected.NegativeLogLikelihoods.Count; index++)
        {
            await Assert.That(Math.Abs(actual.NegativeLogLikelihoods[index] - expected.NegativeLogLikelihoods[index]))
                .IsLessThanOrEqualTo(0.01);
        }
    }
}
