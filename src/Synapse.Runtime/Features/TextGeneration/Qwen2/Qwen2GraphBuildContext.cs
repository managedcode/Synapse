using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed class Qwen2GraphBuildContext
{
    private static readonly NumericType Float = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);
    private uint _nextNode = 1;
    private uint _nextValue = 1;
    private uint _nextTensor = 1;
    private uint _nextState = 1;
    private uint _nextEffect = 1;
    private uint _nextRegion = 1;

    public List<GraphValue> Values { get; } = [];

    public List<GraphNode> Nodes { get; } = [];

    public List<StateSlotDescriptor> StateSlots { get; } = [];

    public List<RegionDescriptor> Regions { get; } = [];

    public List<WeightDescriptor> Weights { get; } = [];

    public GraphValue AddInput(TensorShape shape, NumericType numericType)
    {
        var value = AddValue(shape, numericType);
        _ = AddNode(GraphOperationKind.Input, inputs: null, [value.Id]);
        return value;
    }

    public GraphValue Emit(
        GraphOperationKind operation,
        IEnumerable<ValueId> inputs,
        TensorShape outputShape,
        NumericType outputType,
        ICollection<NodeId> regionNodes,
        IEnumerable<StateSlotId>? stateReads = null,
        IEnumerable<EffectToken>? effectInputs = null,
        GraphOperationAttributes? attributes = null)
    {
        var output = AddValue(outputShape, outputType);
        var node = AddNode(
            operation,
            inputs,
            [output.Id],
            stateReads: stateReads,
            effectInputs: effectInputs,
            attributes: attributes);
        regionNodes.Add(node);
        return output;
    }

    public void EmitStateAppend(
        IEnumerable<ValueId> inputs,
        IEnumerable<StateSlotId> stateWrites,
        EffectToken effect,
        ICollection<NodeId> regionNodes)
    {
        var node = AddNode(
            GraphOperationKind.StateAppend,
            inputs,
            outputs: null,
            stateWrites: stateWrites,
            effectOutputs: [effect]);
        regionNodes.Add(node);
    }

    public WeightValue AddWeight(
        string sourceFile,
        GgufTensorInfo tensor,
        ICollection<NodeId> regionNodes,
        ICollection<TensorId> regionWeights)
    {
        var logicalShape = ToLogicalShape(tensor);
        var value = AddValue(logicalShape, ToNumericType(tensor));
        var tensorId = new TensorId(_nextTensor++);
        Weights.Add(new WeightDescriptor(
            tensorId,
            new WeightSourceRange(sourceFile, tensor.Offset, tensor.ByteLength),
            ToEncoding(tensor),
            logicalShape));
        var node = AddNode(GraphOperationKind.Constant, inputs: null, [value.Id], tensor: tensorId);
        regionNodes.Add(node);
        regionWeights.Add(tensorId);
        return new WeightValue(tensorId, value);
    }

    public StateSlotId AddState(TensorShape shape)
    {
        var id = new StateSlotId(_nextState++);
        StateSlots.Add(new StateSlotDescriptor(id, shape, Float, HasInitialValue: true));
        return id;
    }

    public EffectToken AddEffect() => new(_nextEffect++);

    public void AddRegion(
        IEnumerable<NodeId> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        IEnumerable<TensorId> weights,
        IEnumerable<StateSlotId>? stateReads,
        IEnumerable<StateSlotId>? stateWrites,
        params string[] annotations) => Regions.Add(new RegionDescriptor(
            new RegionId(_nextRegion++),
            nodes,
            inputs,
            outputs,
            weights,
            stateReads,
            stateWrites,
            new RegionActivation(
                new AlwaysActive(),
                new StructuralProvenance(),
                new NotSkippable()),
            annotations));

    public NodeId AddOutput(ValueId value) =>
        AddNode(GraphOperationKind.Output, [value], outputs: null);

    private GraphValue AddValue(TensorShape shape, NumericType numericType)
    {
        var value = new GraphValue(new ValueId(_nextValue++), shape, numericType);
        Values.Add(value);
        return value;
    }

    private NodeId AddNode(
        GraphOperationKind operation,
        IEnumerable<ValueId>? inputs,
        IEnumerable<ValueId>? outputs,
        IEnumerable<StateSlotId>? stateReads = null,
        IEnumerable<StateSlotId>? stateWrites = null,
        IEnumerable<EffectToken>? effectInputs = null,
        IEnumerable<EffectToken>? effectOutputs = null,
        TensorId? tensor = null,
        GraphOperationAttributes? attributes = null)
    {
        var id = new NodeId(_nextNode++);
        Nodes.Add(new GraphNode(
            id,
            operation,
            inputs,
            outputs,
            stateReads,
            stateWrites,
            effectInputs,
            effectOutputs,
            tensor: tensor,
            attributes: attributes));
        return id;
    }

    private static TensorShape ToLogicalShape(GgufTensorInfo tensor) => tensor.Dimensions.Length switch
    {
        1 => new TensorShape(ShapeDimension.Fixed(checked((long)tensor.Dimensions[0]))),
        2 => new TensorShape(
            ShapeDimension.Fixed(checked((long)tensor.Dimensions[1])),
            ShapeDimension.Fixed(checked((long)tensor.Dimensions[0]))),
        _ => throw new NotSupportedException($"Tensor '{tensor.Name}' rank is not supported by Qwen2 graph lowering."),
    };

    private static NumericType ToNumericType(GgufTensorInfo tensor) => tensor.Type switch
    {
        0 => Float,
        8 => new NumericType(StorageDataType.BlockQ8, ComputeDataType.Fp32, AccumulatorDataType.Fp32),
        _ => throw new NotSupportedException($"Tensor '{tensor.Name}' uses unsupported GGUF type {tensor.Type}."),
    };

    internal static WeightEncoding ToEncoding(GgufTensorInfo tensor) => tensor.Type switch
    {
        0 => WeightEncoding.Fp32,
        8 => WeightEncoding.GgmlQ8Zero,
        _ => throw new NotSupportedException($"Tensor '{tensor.Name}' uses unsupported GGUF type {tensor.Type}."),
    };

    public readonly record struct WeightValue(TensorId TensorId, GraphValue Value);
}
