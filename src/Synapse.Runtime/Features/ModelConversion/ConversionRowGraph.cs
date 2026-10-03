using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal sealed record ConversionRowWeight(TensorId Id, ConversionTensor Tensor, bool Matrix);

internal sealed record ConversionRowPlan(ModelGraph Graph, Dictionary<string, ValueId> Values, ConversionRowWeight[] Weights);

internal sealed class ConversionRowGraph
{
    private static readonly NumericType Numeric = new(StorageDataType.Fp32, ComputeDataType.Fp32, AccumulatorDataType.Fp64);
    private readonly List<GraphValue> _declarations = [];
    private readonly List<GraphNode> _nodes = [];
    private readonly Dictionary<string, ValueId> _values = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ValueId> _matrices = new(StringComparer.Ordinal);
    private readonly List<WeightDescriptor> _descriptors = [];
    private readonly List<ConversionRowWeight> _weights = [];

    internal static ConversionRowPlan Build(ConversionModel model, Dictionary<string, ConversionDimension[]> shapes)
    {
        var builder = new ConversionRowGraph();
        foreach (var input in model.Graph.Inputs)
        {
            var id = builder.Declare(input.Name, [input.Shape[^1].Maximum]);
            builder.Add(GraphOperationKind.Input, [], [id]);
        }

        builder.Constants(model);
        foreach (var node in model.Graph.Nodes)
        {
            builder.Operation(node, shapes[node.Output][^1].Maximum);
        }

        return builder.Finish(model.Graph);
    }

    private void Constants(ConversionModel model)
    {
        var matrixNames = model.Graph.Nodes.Where(node => node.Operation == "Linear")
            .Select(node => node.Inputs[1]).ToHashSet(StringComparer.Ordinal);
        var vectorNames = model.Graph.Nodes.SelectMany(node => node.Operation == "Linear"
                ? node.Inputs.Where((_, index) => index != 1) : node.Inputs)
            .Concat(model.Graph.Outputs).ToHashSet(StringComparer.Ordinal);
        foreach (var tensor in model.Tensors)
        {
            if (vectorNames.Contains(tensor.Name))
            {
                Constant(tensor, matrix: false);
            }

            if (matrixNames.Contains(tensor.Name))
            {
                Constant(tensor, matrix: true);
            }
        }
    }

    private void Constant(ConversionTensor tensor, bool matrix)
    {
        var dimensions = matrix ? tensor.Shape : [tensor.Shape[^1]];
        var id = Declare(matrix ? null : tensor.Name, dimensions);
        if (matrix)
        {
            _matrices.Add(tensor.Name, id);
        }

        var tensorId = new TensorId(checked((uint)_weights.Count + 1));
        var shape = _declarations[^1].Shape;
        var length = checked((long)ConversionGraphValidation.Count(dimensions) * sizeof(float));
        _descriptors.Add(new(tensorId, new("weights.bin", checked((long)_weights.Count * ConversionGraphValidation.MaximumElements * sizeof(float)), length),
            WeightEncoding.Fp32, shape));
        _weights.Add(new(tensorId, tensor, matrix));
        Add(GraphOperationKind.Constant, [], [id], tensorId);
    }

    private void Operation(ConversionNode node, long width)
    {
        if (node.Operation == "Identity")
        {
            _values.Add(node.Output, _values[node.Inputs[0]]);
            return;
        }

        var output = Declare(node.Output, [width]);
        var inputs = node.Inputs.Select((name, index) => node.Operation == "Linear" && index == 1
            ? _matrices[name] : _values[name]).ToArray();
        var operation = node.Operation switch
        {
            "Linear" => GraphOperationKind.Linear,
            "Add" => GraphOperationKind.Add,
            "Multiply" => GraphOperationKind.Multiply,
            "Silu" => GraphOperationKind.Silu,
            "Softmax" => GraphOperationKind.Softmax,
            _ => throw new NotSupportedException($"Unsupported prepared operation '{node.Operation}'."),
        };
        Add(operation, inputs, [output]);
    }

    private ConversionRowPlan Finish(ConversionGraph source)
    {
        var inputIds = source.Inputs.Select(input => _values[input.Name]).ToArray();
        var outputIds = source.Outputs.Select(output => _values[output]).Distinct().ToArray();
        if (_nodes.All(node => node.Operation == GraphOperationKind.Input))
        {
            // A constant-only region validates a pure public input alias without introducing arithmetic.
            Constant(new ConversionTensor("identity_anchor", [1], [0]), matrix: true);
        }

        var members = _nodes.Where(node => node.Operation != GraphOperationKind.Input).ToArray();
        var produced = members.SelectMany(node => node.Outputs).ToHashSet();
        var regionInputs = members.SelectMany(node => node.Inputs).Where(id => !produced.Contains(id)).Distinct().ToArray();
        var regionOutputs = outputIds.Where(produced.Contains).ToArray();
        Add(GraphOperationKind.Output, outputIds, []);
        var region = new RegionDescriptor(new RegionId(1), members.Select(node => node.Id), regionInputs, regionOutputs,
            _weights.Select(weight => weight.Id), [], [],
            new(new AlwaysActive(), new StructuralProvenance(), new NotSkippable()));
        var graph = new ModelGraph(new GraphVersion(1, 0), new OpSetVersion(1, 0), _declarations, _nodes, [],
            [new GraphEntryPoint(new EntryPointId(1), "forward", inputIds, outputIds)], [region], _descriptors);
        return new(graph, _values, [.. _weights]);
    }

    private ValueId Declare(string? name, long[] shape)
    {
        var id = new ValueId(checked((uint)_declarations.Count + 1));
        _declarations.Add(new(id, new TensorShape([.. shape.Select(ShapeDimension.Fixed)]), Numeric));
        if (name is not null)
        {
            _values.Add(name, id);
        }

        return id;
    }

    private void Add(GraphOperationKind kind, ValueId[] inputs, ValueId[] outputs, TensorId? tensor = null) =>
        _nodes.Add(new GraphNode(new NodeId(checked((uint)_nodes.Count + 1)), kind, inputs, outputs, tensor: tensor));
}
