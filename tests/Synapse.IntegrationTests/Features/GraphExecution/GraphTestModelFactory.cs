using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

internal static class GraphTestModelFactory
{
    private static readonly NumericType Fp32 = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);

    public static ModelGraph CreateLinearGraph(
        TensorShape outputShape,
        ValueId? conditionalPredicate = null,
        NumericType? numericType = null)
    {
        var numeric = numericType ?? Fp32;
        var input = new GraphValue(
            new ValueId(1),
            new TensorShape(ShapeDimension.Fixed(4)),
            numeric);
        var weights = new GraphValue(
            new ValueId(2),
            new TensorShape(ShapeDimension.Fixed(3), ShapeDimension.Fixed(4)),
            numeric);
        var output = new GraphValue(new ValueId(3), outputShape, numeric);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [input.Id]),
            new GraphNode(
                new NodeId(2),
                GraphOperationKind.Constant,
                outputs: [weights.Id],
                tensor: new TensorId(1)),
            new GraphNode(new NodeId(3), GraphOperationKind.Linear, [input.Id, weights.Id], [output.Id]),
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [output.Id]),
        };
        return CreateGraph(
            [input, weights, output],
            nodes,
            [input.Id],
            [output.Id],
            activation: conditionalPredicate is { } predicate
                ? new RegionActivation(
                    new PredicateDecision(predicate, RouteScope.Step),
                    new ProgrammedProvenance(),
                    new OutputsAbsent())
                : AlwaysStructural());
    }

    public static ModelGraph CreateGraph(
        IEnumerable<GraphValue> values,
        IEnumerable<GraphNode> nodes,
        IReadOnlyList<ValueId> inputs,
        IReadOnlyList<ValueId> outputs,
        IEnumerable<StateSlotDescriptor>? stateSlots = null,
        RegionActivation? activation = null)
    {
        var valueArray = values.ToArray();
        var nodeArray = nodes.ToArray();
        var memberNodes = nodeArray
            .Where(node => node.Operation is not (GraphOperationKind.Input or GraphOperationKind.Output))
            .ToArray();
        var region = CreateRegion(nodeArray, memberNodes, outputs, activation);
        var valuesById = valueArray.ToDictionary(value => value.Id);
        var weights = memberNodes
            .Where(node => node.Operation == GraphOperationKind.Constant)
            .Select(node => CreateWeight(node, valuesById))
            .ToArray();
        return new ModelGraph(
            new GraphVersion(1, 0),
            new OpSetVersion(1, 0),
            valueArray,
            nodeArray,
            stateSlots,
            [new GraphEntryPoint(new EntryPointId(1), "forward", inputs, outputs)],
            [region],
            weights);
    }

    public static ModelGraph ReplaceRegion(ModelGraph graph, RegionDescriptor region) => new(
        graph.GraphVersion,
        graph.OpSetVersion,
        graph.Values,
        graph.Nodes,
        graph.StateSlots,
        graph.EntryPoints,
        [region],
        graph.Weights);

    private static RegionDescriptor CreateRegion(
        IReadOnlyList<GraphNode> allNodes,
        IReadOnlyList<GraphNode> memberNodes,
        IReadOnlyList<ValueId> outputs,
        RegionActivation? activation)
    {
        var memberIds = memberNodes.Select(node => node.Id).ToHashSet();
        var producers = allNodes
            .SelectMany(node => node.Outputs.Select(output => (output, node.Id)))
            .ToDictionary(pair => pair.output, pair => pair.Id);
        var inputs = memberNodes
            .SelectMany(node => node.Inputs)
            .Where(input => !producers.TryGetValue(input, out var producer) || !memberIds.Contains(producer))
            .Distinct();
        var regionOutputs = memberNodes
            .SelectMany(node => node.Outputs)
            .Where(output => outputs.Contains(output) || allNodes.Any(node =>
                !memberIds.Contains(node.Id) && node.Inputs.Contains(output)))
            .Distinct();
        return new RegionDescriptor(
            new RegionId(1),
            memberNodes.Select(node => node.Id),
            inputs,
            regionOutputs,
            memberNodes.Where(node => node.Operation == GraphOperationKind.Constant)
                .Select(node => node.Tensor!.Value),
            memberNodes.SelectMany(node => node.StateReads).Distinct(),
            memberNodes.SelectMany(node => node.StateWrites).Distinct(),
            activation ?? AlwaysStructural(),
            semanticAnnotations: ["test"]);
    }

    private static RegionActivation AlwaysStructural() => new(
        new AlwaysActive(),
        new StructuralProvenance(),
        new NotSkippable());

    private static WeightDescriptor CreateWeight(
        GraphNode node,
        Dictionary<ValueId, GraphValue> values)
    {
        var tensorId = node.Tensor!.Value;
        var value = values[node.Outputs.Single()];
        var encoding = value.NumericType.Storage switch
        {
            StorageDataType.Fp32 => WeightEncoding.Fp32,
            StorageDataType.BlockQ8 => WeightEncoding.GgmlQ8Zero,
            StorageDataType.Fp16 => throw new NotImplementedException(),
            StorageDataType.Bf16 => throw new NotImplementedException(),
            StorageDataType.BlockQ4 => throw new NotImplementedException(),
            StorageDataType.I32 => throw new NotImplementedException(),
            StorageDataType.Bool => throw new NotImplementedException(),
            _ => throw new InvalidOperationException("Test graph weight uses an unsupported encoding."),
        };
        return new WeightDescriptor(
            tensorId,
            new WeightSourceRange("test.gguf", checked((long)tensorId.Value * 4096), 4096),
            encoding,
            value.Shape);
    }
}
