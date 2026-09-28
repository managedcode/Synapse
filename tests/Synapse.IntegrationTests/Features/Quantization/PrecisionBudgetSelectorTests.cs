using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class PrecisionBudgetSelectorTests
{
    private const string Source = "gguf.q8_0";

    [Test]
    public async Task FullBudgetKeepsSourcePrecision()
    {
        var plan = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(10_000, AllowApproximation: false));

        await Assert.That(plan.IsApproximate).IsFalse();
        await Assert.That(plan.TotalBytes).IsEqualTo(300L);
        await Assert.That(plan.Assignments.All(item => item.EncodingId == Source)).IsTrue();
    }

    [Test]
    public async Task LeastImportantTensorIsDemotedFirst()
    {
        var plan = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(250, AllowApproximation: true));

        var demoted = plan.Assignments.Where(item => item.IsDemoted).Select(item => item.Tensor).ToArray();
        await Assert.That(demoted).IsEquivalentTo([new TensorId(2)]);
        await Assert.That(plan.TotalBytes).IsLessThanOrEqualTo(250L);
        await Assert.That(plan.IsApproximate).IsTrue();
    }

    [Test]
    public async Task UnimportantTensorsCanDropToTernaryWhileImportantStayHigh()
    {
        var plan = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(180, AllowApproximation: true));

        var byTensor = plan.Assignments.ToDictionary(item => item.Tensor);
        await Assert.That(byTensor[new TensorId(1)].EncodingId).IsEqualTo(Source);
        await Assert.That(byTensor[new TensorId(2)].EncodingId).IsEqualTo(TernaryBlockCodec.EncodingId);
        await Assert.That(plan.TotalBytes).IsLessThanOrEqualTo(180L);
    }

    [Test]
    public async Task PinnedTensorsStayAtSourcePrecision()
    {
        var candidates = Candidates().Select(item => item.Tensor == new TensorId(2) ? item with { PinnedHigh = true } : item).ToArray();

        var plan = PrecisionBudgetSelector.Select(candidates, new PrecisionPolicy(250, AllowApproximation: true));

        var tensor2 = plan.Assignments.Single(item => item.Tensor == new TensorId(2));
        await Assert.That(tensor2.EncodingId).IsEqualTo(Source);
    }

    [Test]
    public async Task InfeasibleBudgetRejectedWithRequirement()
    {
        var failure = await Assert.That(() => PrecisionBudgetSelector.Select(
            Candidates(), new PrecisionPolicy(50, AllowApproximation: true))).Throws<PrecisionBudgetException>();

        await Assert.That(failure!.Failure).IsEqualTo(PrecisionBudgetFailure.InsufficientBudget);
        await Assert.That(failure.MinimumBytes).IsEqualTo(75L);
    }

    [Test]
    public async Task ApproximationRequiresExplicitPolicy()
    {
        var failure = await Assert.That(() => PrecisionBudgetSelector.Select(
            Candidates(), new PrecisionPolicy(250, AllowApproximation: false))).Throws<PrecisionBudgetException>();

        await Assert.That(failure!.Failure).IsEqualTo(PrecisionBudgetFailure.ApproximationNotAllowed);
    }

    [Test]
    public async Task DeviceWithoutLowPrecisionKernelNeverReceivesIt()
    {
        var policy = new PrecisionPolicy(
            250,
            AllowApproximation: true,
            SupportedEncodings: new HashSet<string>(StringComparer.Ordinal) { Source, SynQ4BlockCodec.EncodingId });

        var plan = PrecisionBudgetSelector.Select(Candidates(), policy);

        await Assert.That(plan.Assignments.Any(item => item.EncodingId == TernaryBlockCodec.EncodingId)).IsFalse();
    }

    [Test]
    public async Task ProfileHashIdentifiesEffectiveWeightsOnly()
    {
        var first = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(250, AllowApproximation: true));
        var reordered = PrecisionBudgetSelector.Select(
            [.. Candidates().Reverse()], new PrecisionPolicy(260, AllowApproximation: true));
        var different = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(180, AllowApproximation: true));

        await Assert.That(reordered.ProfileHash).IsEqualTo(first.ProfileHash);
        await Assert.That(different.ProfileHash).IsNotEqualTo(first.ProfileHash);
        await Assert.That(first.ProfileHash.Value.Length).IsEqualTo(64);
    }

    [Test]
    public async Task EqualScoresBreakTiesByTensorId()
    {
        PrecisionCandidate[] candidates =
        [
            Candidate(7, 0.1, 0.5),
            Candidate(3, 0.1, 0.5),
        ];

        var plan = PrecisionBudgetSelector.Select(candidates, new PrecisionPolicy(160, AllowApproximation: true));

        var demoted = plan.Assignments.Single(item => item.IsDemoted);
        await Assert.That(demoted.Tensor).IsEqualTo(new TensorId(3));
    }

    [Test]
    public async Task LargerBudgetOnlyPromotes()
    {
        long[] budgets = [75, 100, 150, 180, 200, 225, 250, 280, 300];
        var plans = budgets
            .Select(budget => PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(budget, AllowApproximation: true)))
            .ToArray();

        for (var index = 1; index < plans.Length; index++)
        {
            var smaller = plans[index - 1].Assignments.ToDictionary(item => item.Tensor, item => item.Bytes);
            var larger = plans[index].Assignments;
            await Assert.That(larger.All(item => item.Bytes >= smaller[item.Tensor])).IsTrue();
        }
    }

    [Test]
    public async Task PromotionRestoresMostImportantDemotedTensorsFirst()
    {
        var tight = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(150, AllowApproximation: true));
        var relaxed = PrecisionBudgetSelector.Select(Candidates(), new PrecisionPolicy(225, AllowApproximation: true));

        var promotions = PrecisionBudgetSelector.Diff(tight, relaxed);
        var demotions = PrecisionBudgetSelector.Diff(relaxed, tight);

        await Assert.That(promotions.Select(change => change.Tensor.Value)).IsEquivalentTo([1u, 3u]);
        await Assert.That(promotions.All(change =>
            change.IsPromotion && change.FromEncoding == SynQ4BlockCodec.EncodingId && change.ToEncoding == Source)).IsTrue();
        await Assert.That(demotions.All(change => !change.IsPromotion)).IsTrue();
        await Assert.That(promotions.Sum(change => change.BytesDelta)).IsEqualTo(relaxed.TotalBytes - tight.TotalBytes);
    }

    private static PrecisionCandidate[] Candidates() =>
    [
        Candidate(1, 0.20, 0.90),
        Candidate(2, 0.01, 0.05),
        Candidate(3, 0.10, 0.60),
    ];

    private static PrecisionCandidate Candidate(uint tensor, double q4Distortion, double ternaryDistortion) =>
        new(
            new TensorId(tensor),
            [
                new PrecisionOption(Source, 100, 0),
                new PrecisionOption(SynQ4BlockCodec.EncodingId, 50, q4Distortion),
                new PrecisionOption(TernaryBlockCodec.EncodingId, 25, ternaryDistortion),
            ]);
}
