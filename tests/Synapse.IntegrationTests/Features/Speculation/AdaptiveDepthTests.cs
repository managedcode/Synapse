using ManagedCode.Synapse.Runtime.Features.Speculation;

namespace ManagedCode.Synapse.IntegrationTests.Features.Speculation;

/// <summary>TEST-SPC-005-1: numerical controller contracts using the production cost estimator.</summary>
public sealed class AdaptiveDepthTests
{
    [Test]
    public async Task SelectsThroughputRatherThanAcceptanceOrLongestDraft()
    {
        var policy = new AdaptiveDepth(3);
        for (var sample = 0; sample < 2; sample++)
        {
            for (var depth = 0; depth <= 3; depth++)
            {
                await Assert.That(policy.Select(3)).IsEqualTo(depth);
                policy.Observe(depth, depth, depth + 1, depth * 0.5, depth < 2 ? 1 : 10);
            }
        }

        await Assert.That(policy.Select(3)).IsEqualTo(1);
        await Assert.That(policy.Select(0)).IsEqualTo(0);
    }

    [Test]
    public async Task PlainWinsUnlessDraftPaysAndReprobesStayBounded()
    {
        var policy = new AdaptiveDepth(1);
        for (var sample = 0; sample < 2; sample++)
        {
            policy.Observe(0, 0, 1, 0, 1);
            policy.Observe(1, 1, 2, 1, 1);
        }

        for (var round = 0; round < 16; round++)
        {
            await Assert.That(policy.Select(1)).IsEqualTo(0);
        }

        await Assert.That(policy.Select(1)).IsEqualTo(1);
        for (var sample = 0; sample < 8; sample++)
        {
            policy.Observe(1, 1, 2, 0.1, 0.1);
        }

        await Assert.That(policy.Select(1)).IsEqualTo(1);
        var measurements = policy.Measurements;
        await Assert.That(measurements.Sum(row => row.Rounds)).IsEqualTo(12);
        await Assert.That(measurements[1].CommittedTokens).IsEqualTo(20);
    }

    [Test]
    public async Task ClippedRoundsAreReportedWithoutBiasingTheEstimator()
    {
        var policy = new AdaptiveDepth(1);
        policy.Observe(0, 0, 1, 0, 1);
        policy.Observe(1, 1, 2, 0, 1);
        policy.Observe(0, 0, 1, 0, 1);
        policy.Observe(1, 1, 2, 0, 1);
        policy.Observe(1, 0, 1, 0, 100, learn: false);
        await Assert.That(policy.Select(1)).IsEqualTo(1);
        await Assert.That(policy.Measurements[1].Rounds).IsEqualTo(3);
        await Assert.That(policy.Measurements[1].VerificationMilliseconds).IsEqualTo(102);
        for (var sample = 0; sample < 8; sample++)
        {
            policy.Observe(1, 0, 1, 2, 10);
        }

        await Assert.That(policy.Select(1)).IsEqualTo(0);
    }

    [Test]
    public async Task SmallApparentGainsDoNotTriggerDrafting()
    {
        var policy = new AdaptiveDepth(1);
        for (var sample = 0; sample < 2; sample++)
        {
            policy.Observe(0, 0, 1, 0, 1);
            policy.Observe(1, 1, 2, 0, 2 / 1.04);
        }

        await Assert.That(policy.Select(1)).IsEqualTo(0);
    }

    [Test]
    public async Task InvalidMeasurementsFailExplicitly()
    {
        var policy = new AdaptiveDepth(2);
        await Assert.That(() => new AdaptiveDepth(0)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => new AdaptiveDepth(8)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(3, 0, 1, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 2, 1, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 1, 0, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 1, 1, double.NaN, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 1, 1, 0, double.PositiveInfinity)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Select(-1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(-1, 0, 1, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, -1, 1, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 1, 3, 0, 1)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => policy.Observe(1, 1, 1, -1, 1)).Throws<ArgumentOutOfRangeException>();
    }
}
