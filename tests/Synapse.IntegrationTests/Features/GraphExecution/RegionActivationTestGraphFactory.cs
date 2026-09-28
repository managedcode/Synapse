using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using static ManagedCode.Synapse.IntegrationTests.Features.GraphExecution.GraphTestModelFactory;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

internal static class RegionActivationTestGraphFactory
{
    private static readonly NumericType Fp32 = new(
        StorageDataType.Fp32, ComputeDataType.Fp32, AccumulatorDataType.Fp32);
    private static readonly NumericType I32 = new(
        StorageDataType.I32, ComputeDataType.I32, AccumulatorDataType.I32);
    private static readonly NumericType Boolean = new(
        StorageDataType.Bool, ComputeDataType.Bool, AccumulatorDataType.I32);

    public static ModelGraph CreateRouteGraph(
        TopKRouteAxis axis = TopKRouteAxis.Sequence,
        bool routeAfterRegion = false)
    {
        var data = new GraphValue(new ValueId(1), Vector(4), Fp32);
        var route = new GraphValue(new ValueId(2), Vector(1), I32);
        var output = new GraphValue(new ValueId(3), Vector(4), Fp32);
        var routeNode = new GraphNode(
            new NodeId(2),
            GraphOperationKind.TopKRoute,
            [data.Id],
            [route.Id],
            attributes: new TopKRouteAttributes(
                1, axis, RouteTiePolicy.StableLowestIndex, Capacity: null));
        var regionNode = new GraphNode(new NodeId(3), GraphOperationKind.Silu, [data.Id], [output.Id]);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [data.Id]),
            routeAfterRegion ? regionNode : routeNode,
            routeAfterRegion ? routeNode : regionNode,
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [output.Id]),
        };
        var routeRegion = Region(
            1, [new NodeId(2)], [data.Id], [route.Id], AlwaysStructural());
        var conditionalRegion = Region(
            2, [new NodeId(3)], [data.Id], [output.Id],
            new RegionActivation(
                new RouteSlotDecision(route.Id, 0, RouteScope.Step),
                new ProgrammedProvenance(),
                new BypassOutputs([new ValueBypass(output.Id, data.Id)])));
        return Graph([data, route, output], nodes, [data.Id], [output.Id],
            [routeRegion, conditionalRegion]);
    }

    public static ModelGraph CreateAbsentOutputGraph()
    {
        var data = new GraphValue(new ValueId(1), Vector(4), Fp32);
        var intermediate = new GraphValue(new ValueId(2), Vector(4), Fp32);
        var output = new GraphValue(new ValueId(3), Vector(4), Fp32);
        var predicate = new GraphValue(new ValueId(4), Vector(1), Boolean);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [data.Id]),
            new GraphNode(new NodeId(2), GraphOperationKind.Input, outputs: [predicate.Id]),
            new GraphNode(new NodeId(3), GraphOperationKind.Silu, [data.Id], [intermediate.Id]),
            new GraphNode(new NodeId(4), GraphOperationKind.Gelu, [intermediate.Id], [output.Id]),
            new GraphNode(new NodeId(5), GraphOperationKind.Output, inputs: [output.Id]),
        };
        var optional = Region(
            1, [new NodeId(3)], [data.Id], [intermediate.Id],
            new RegionActivation(
                new PredicateDecision(predicate.Id, RouteScope.Step),
                new ProgrammedProvenance(),
                new OutputsAbsent()));
        var consumer = Region(
            2, [new NodeId(4)], [intermediate.Id], [output.Id], AlwaysStructural());
        return Graph(
            [data, intermediate, output, predicate], nodes,
            [data.Id, predicate.Id], [output.Id], [optional, consumer]);
    }

    public static ModelGraph CreateSkippableStateWriterGraph()
    {
        var data = new GraphValue(new ValueId(1), Vector(4), Fp32);
        var position = new GraphValue(new ValueId(2), Vector(1), I32);
        var predicate = new GraphValue(new ValueId(3), Vector(1), Boolean);
        var slot = new StateSlotDescriptor(
            new StateSlotId(1),
            new TensorShape(ShapeDimension.Bounded("Context", 1, 8), ShapeDimension.Fixed(4)),
            Fp32,
            HasInitialValue: true,
            PositionHolesAllowed: false);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [data.Id]),
            new GraphNode(new NodeId(2), GraphOperationKind.Input, outputs: [position.Id]),
            new GraphNode(new NodeId(3), GraphOperationKind.Input, outputs: [predicate.Id]),
            new GraphNode(
                new NodeId(4), GraphOperationKind.StateAppend,
                [data.Id, position.Id], stateWrites: [slot.Id]),
            new GraphNode(new NodeId(5), GraphOperationKind.Output, inputs: [data.Id]),
        };
        var region = Region(
            1, [new NodeId(4)], [data.Id, position.Id], [],
            new RegionActivation(
                new PredicateDecision(predicate.Id, RouteScope.Step),
                new ProgrammedProvenance(),
                new OutputsAbsent()),
            stateWrites: [slot.Id]);
        return Graph(
            [data, position, predicate], nodes,
            [data.Id, position.Id, predicate.Id], [data.Id], [region], [slot]);
    }

    public static ModelGraph WithActivation(ModelGraph graph, RegionActivation activation)
    {
        var source = graph.Regions.Single();
        var replacement = Region(
            source.Id.Value, source.Nodes, source.Inputs, source.Outputs,
            activation, source.RequiredWeights, source.StateReads,
            source.StateWrites, source.SemanticAnnotations);
        return ReplaceRegion(graph, replacement);
    }

    public static TensorShape Vector(long size) => new(ShapeDimension.Fixed(size));

    public static ContentHash Hash(char character) => new(new string(character, 64));

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

    private static RegionDescriptor Region(
        uint id,
        IEnumerable<NodeId> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        RegionActivation activation,
        IEnumerable<TensorId>? weights = null,
        IEnumerable<StateSlotId>? stateReads = null,
        IEnumerable<StateSlotId>? stateWrites = null,
        IEnumerable<string>? annotations = null) => new(
        new RegionId(id), nodes, inputs, outputs, weights,
        stateReads, stateWrites, activation, annotations);

    private static RegionActivation AlwaysStructural() => new(
        new AlwaysActive(), new StructuralProvenance(), new NotSkippable());
}
