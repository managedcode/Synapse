using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ExecutionPlanning;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.ExecutionPlanning;

public sealed class HeterogeneousPlacementPlannerTests
{
    private const string Source = "gguf.q8_0";
    private const long Unlimited = 1_000_000_000;
    private static readonly LinkProfile FastLink = new(LatencySeconds: 0.001, BytesPerSecond: 1_000_000);

    [Test]
    public async Task ThroughputObjectiveGivesFasterDeviceMoreLayers()
    {
        var plan = HeterogeneousPlacementPlanner.Plan(Request(
            [Device("fast", Unlimited, 3000), Device("slow", Unlimited, 1000)],
            PlacementObjective.Throughput));

        var layers = plan.Stages.ToDictionary(stage => stage.DeviceId, stage => stage.LastLayer - stage.FirstLayer + 1);
        await Assert.That(layers["fast"]).IsEqualTo(9);
        await Assert.That(layers["slow"]).IsEqualTo(3);
        await Assert.That(Math.Abs(plan.EstimatedTokenSeconds - 0.3)).IsLessThan(1e-9);
    }

    [Test]
    public async Task LatencyObjectiveKeepsSequenceOnFastestDevice()
    {
        var plan = HeterogeneousPlacementPlanner.Plan(Request(
            [Device("fast", Unlimited, 3000), Device("slow", Unlimited, 1000)],
            PlacementObjective.Latency));

        var stage = plan.Stages.Single();
        await Assert.That(stage.DeviceId).IsEqualTo("fast");
        await Assert.That(stage.FirstLayer).IsEqualTo(0);
        await Assert.That(stage.LastLayer).IsEqualTo(11);
    }

    [Test]
    public async Task UnqualifiedQualityPlanRejected()
    {
        var plan = HeterogeneousPlacementPlanner.Plan(Request(
            [Device("fast-small", 900, 3000), Device("slow-large", Unlimited, 500)],
            PlacementObjective.Latency,
            allowApproximation: false));

        await Assert.That(plan.IsApproximate).IsFalse();
        await Assert.That(plan.Stages.SelectMany(stage => stage.Precision.Assignments)
            .All(item => item.EncodingId == Source)).IsTrue();
        await Assert.That(plan.Stages.All(stage => stage.ReservedBytes <= stage.MemoryBudgetBytes)).IsTrue();
    }

    [Test]
    public async Task SmallFastDeviceDemotesLeastImportantTensorsWhenAllowed()
    {
        var plan = HeterogeneousPlacementPlanner.Plan(Request(
            [Device("fast-small", 900, 3000), Device("slow-large", Unlimited, 500)],
            PlacementObjective.Latency,
            allowApproximation: true));

        var stage = plan.Stages.Single();
        var demoted = stage.Precision.Assignments.Where(item => item.IsDemoted).Select(item => item.Tensor.Value).ToArray();
        await Assert.That(stage.DeviceId).IsEqualTo("fast-small");
        await Assert.That(plan.IsApproximate).IsTrue();
        await Assert.That(demoted).IsEquivalentTo([0u, 1u, 2u, 3u, 4u, 5u]);
        await Assert.That(Math.Abs(plan.EstimatedTokenSeconds - 0.3)).IsLessThan(1e-9);
    }

    [Test]
    public async Task KvReservationCounted()
    {
        var request = Request([Device("only", 800, 3000)], PlacementObjective.Latency, allowApproximation: true) with
        {
            ContextTokens = 10,
        };

        var failure = await Assert.That(() => HeterogeneousPlacementPlanner.Plan(request))
            .Throws<PlacementException>();

        await Assert.That(failure!.Failure).IsEqualTo(PlacementFailure.InsufficientMemory);
    }

    [Test]
    public async Task SlowLinkKeepsModelOnOneDevice()
    {
        var request = Request(
            [Device("fast", Unlimited, 3000), Device("slow", Unlimited, 1000)],
            PlacementObjective.Throughput) with
        {
            Link = new LinkProfile(LatencySeconds: 10, BytesPerSecond: 1_000_000),
        };

        var plan = HeterogeneousPlacementPlanner.Plan(request);

        await Assert.That(plan.Stages.Single().DeviceId).IsEqualTo("fast");
    }

    [Test]
    public async Task SearchIsBoundedToFourDevices()
    {
        var devices = Enumerable.Range(0, 5).Select(index => Device($"d{index}", Unlimited, 1000)).ToArray();

        await Assert.That(() => HeterogeneousPlacementPlanner.Plan(Request(devices, PlacementObjective.Latency)))
            .Throws<ArgumentException>();
    }

    private static PlacementRequest Request(
        DeviceProfile[] devices,
        PlacementObjective objective,
        bool allowApproximation = false) =>
        new(
            Layers(),
            devices,
            FastLink,
            ActivationBytesPerToken: 10,
            ContextTokens: 0,
            allowApproximation,
            objective);

    private static PlacementLayer[] Layers() =>
    [
        .. Enumerable.Range(0, 12).Select(index => new PlacementLayer(
            index,
            [
                new PrecisionCandidate(
                    new TensorId((uint)index),
                    [
                        new PrecisionOption(Source, 100, 0),
                        new PrecisionOption(SynQ4BlockCodec.EncodingId, 50, (index + 1) * 0.01),
                        new PrecisionOption(TernaryBlockCodec.EncodingId, 25, (index + 1) * 0.1),
                    ]),
            ],
            KvBytesPerToken: 5)),
    ];

    private static DeviceProfile Device(string id, long memory, double bytesPerSecond) =>
        new(id, memory, bytesPerSecond, LayerOverheadSeconds: 0);
}
