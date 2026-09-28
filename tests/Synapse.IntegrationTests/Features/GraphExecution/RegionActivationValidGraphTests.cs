using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class RegionActivationValidGraphTests
{
    private static readonly NumericType Fp32 = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);
    private static readonly NumericType I32 = new(
        StorageDataType.I32,
        ComputeDataType.I32,
        AccumulatorDataType.I32);
    private static readonly NumericType Boolean = new(
        StorageDataType.Bool,
        ComputeDataType.Bool,
        AccumulatorDataType.I32);

    [Test]
    public async Task FeatureRouteCanSelectMatchingBypass()
    {
        var data = Value(1, Fp32, 4);
        var route = Value(2, I32, 1);
        var output = Value(3, Fp32, 4);
        var nodes = new[]
        {
            Input(1, data),
            new GraphNode(
                new NodeId(2),
                GraphOperationKind.TopKRoute,
                [data.Id],
                [route.Id],
                attributes: new TopKRouteAttributes(1, TopKRouteAxis.Feature, RouteTiePolicy.StableLowestIndex, null)),
            new GraphNode(new NodeId(3), GraphOperationKind.Silu, [data.Id], [output.Id]),
            Output(4, output),
        };
        var regions = new[]
        {
            Region(1, [2], [data.Id], [route.Id], Always()),
            Region(2, [3], [data.Id], [output.Id], new RegionActivation(
                new RouteSlotDecision(route.Id, 0, RouteScope.Step),
                new StructuralProvenance(),
                new BypassOutputs([new ValueBypass(output.Id, data.Id)]))),
        };

        var result = ModelGraphVerifier.Verify(Graph([data, route, output], nodes, [data.Id], [output.Id], regions));

        await Assert.That(result.Diagnostics).IsEmpty();
    }

    [Test]
    public async Task AbsentOutputCanFeedAddMerge()
    {
        var data = Value(1, Fp32, 4);
        var predicate = Value(2, Boolean, 1);
        var optional = Value(3, Fp32, 4);
        var resultValue = Value(4, Fp32, 4);
        var nodes = new[]
        {
            Input(1, data),
            Input(2, predicate),
            new GraphNode(new NodeId(3), GraphOperationKind.Silu, [data.Id], [optional.Id]),
            new GraphNode(
                new NodeId(4),
                GraphOperationKind.Merge,
                [data.Id, optional.Id],
                [resultValue.Id],
                mergeMode: MergeMode.Add),
            Output(5, resultValue),
        };
        var regions = new[]
        {
            Region(1, [3], [data.Id], [optional.Id], new RegionActivation(
                new PredicateDecision(predicate.Id, RouteScope.Step),
                new ProgrammedProvenance(),
                new OutputsAbsent())),
            Region(2, [4], [data.Id, optional.Id], [resultValue.Id], Always()),
        };

        var result = ModelGraphVerifier.Verify(Graph(
            [data, predicate, optional, resultValue],
            nodes,
            [data.Id, predicate.Id],
            [resultValue.Id],
            regions));

        await Assert.That(result.Diagnostics).IsEmpty();
    }

    [Test]
    public async Task PositionHoleSlotRequiresHoleAwareAttention()
    {
        var result = ModelGraphVerifier.Verify(CreateStateRouteGraph(holeAware: false));

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidRegion &&
            item.Message.Contains("hole-aware", StringComparison.Ordinal))).IsTrue();
    }

    [Test]
    public async Task PositionHoleSlotWithHoleAwareAttentionPasses()
    {
        var result = ModelGraphVerifier.Verify(CreateStateRouteGraph(holeAware: true));

        await Assert.That(result.Diagnostics).IsEmpty();
    }

    [Test]
    public async Task ActivationProvenanceChangesModelFingerprint()
    {
        var graph = CreateStateRouteGraph(holeAware: true);
        var region = graph.Regions[0];
        var revised = Region(
            region.Id.Value,
            region.Nodes.Select(node => node.Value),
            region.Inputs,
            region.Outputs,
            new RegionActivation(
                region.Activation.Decision,
                new ApproximateProvenance(new ContentHash(new string('a', 64))),
                region.Activation.Skip),
            region.StateReads,
            region.StateWrites);
        var changed = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes,
            graph.StateSlots,
            graph.EntryPoints,
            [revised, graph.Regions[1]],
            graph.Weights);

        await Assert.That(ModelGraphVerifier.Verify(changed).IsValid).IsTrue();
        await Assert.That(ModelGraphFingerprint.Compute(graph))
            .IsNotEqualTo(ModelGraphFingerprint.Compute(changed));
    }

    private static ModelGraph CreateStateRouteGraph(bool holeAware)
    {
        var data = Value(1, Fp32, 4);
        var position = Value(2, I32, 1);
        var predicate = Value(3, Boolean, 1);
        var output = Value(4, Fp32, 4);
        var stateShape = new TensorShape(ShapeDimension.Bounded("Context", 1, 8), ShapeDimension.Fixed(4));
        var routedSlot = new StateSlotDescriptor(new StateSlotId(1), stateShape, Fp32, true, true);
        var companionSlot = new StateSlotDescriptor(new StateSlotId(2), stateShape, Fp32, true);
        var nodes = new[]
        {
            Input(1, data),
            Input(2, position),
            Input(3, predicate),
            new GraphNode(new NodeId(4), GraphOperationKind.StateAppend,
                [data.Id, position.Id], stateWrites: [routedSlot.Id]),
            new GraphNode(new NodeId(5), GraphOperationKind.CausalAttention,
                [data.Id, position.Id], [output.Id], stateReads: [routedSlot.Id, companionSlot.Id],
                attributes: new CausalAttentionAttributes(
                    1, 1, 4, 0.5f, AttentionMaskKind.Causal, holeAware)),
            Output(6, output),
        };
        var regions = new[]
        {
            Region(1, [4], [data.Id, position.Id], [], new RegionActivation(
                new PredicateDecision(predicate.Id, RouteScope.Step),
                new ProgrammedProvenance(),
                new OutputsAbsent()), stateWrites: [routedSlot.Id]),
            Region(2, [5], [data.Id, position.Id], [output.Id], Always(),
                stateReads: [routedSlot.Id, companionSlot.Id]),
        };
        return Graph([data, position, predicate, output], nodes,
            [data.Id, position.Id, predicate.Id], [output.Id], regions,
            [routedSlot, companionSlot]);
    }

    private static GraphValue Value(uint id, NumericType type, long width) => new(
        new ValueId(id), new TensorShape(ShapeDimension.Fixed(width)), type);

    private static GraphNode Input(uint id, GraphValue value) => new(
        new NodeId(id), GraphOperationKind.Input, outputs: [value.Id]);

    private static GraphNode Output(uint id, GraphValue value) => new(
        new NodeId(id), GraphOperationKind.Output, inputs: [value.Id]);

    private static RegionActivation Always() => new(
        new AlwaysActive(), new StructuralProvenance(), new NotSkippable());

    private static RegionDescriptor Region(
        uint id,
        IEnumerable<uint> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        RegionActivation activation,
        IEnumerable<StateSlotId>? stateReads = null,
        IEnumerable<StateSlotId>? stateWrites = null) => new(
        new RegionId(id),
        nodes.Select(node => new NodeId(node)),
        inputs,
        outputs,
        requiredWeights: null,
        stateReads,
        stateWrites,
        activation);

    private static ModelGraph Graph(
        IEnumerable<GraphValue> values,
        IEnumerable<GraphNode> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        IEnumerable<RegionDescriptor> regions,
        IEnumerable<StateSlotDescriptor>? stateSlots = null) => new(
        new GraphVersion(1, 0),
        new OpSetVersion(1, 0),
        values,
        nodes,
        stateSlots,
        [new GraphEntryPoint(new EntryPointId(1), "forward-token", inputs, outputs)],
        regions);
}
