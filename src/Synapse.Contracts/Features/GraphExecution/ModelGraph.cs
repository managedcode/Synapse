namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Operation semantics supported by model IR version 1.</summary>
public enum GraphOperationKind
{
    /// <summary>Entry-point input.</summary>
    Input,
    /// <summary>Immutable constant.</summary>
    Constant,
    /// <summary>Token embedding lookup.</summary>
    Embedding,
    /// <summary>Root-mean-square normalization.</summary>
    RmsNorm,
    /// <summary>Layer normalization.</summary>
    LayerNorm,
    /// <summary>Dense linear projection.</summary>
    Linear,
    /// <summary>Quantized dense linear projection.</summary>
    QuantizedLinear,
    /// <summary>Element-wise addition.</summary>
    Add,
    /// <summary>Element-wise multiplication.</summary>
    Multiply,
    /// <summary>SiLU activation.</summary>
    Silu,
    /// <summary>GELU activation.</summary>
    Gelu,
    /// <summary>Logical reshape.</summary>
    Reshape,
    /// <summary>Dimension permutation.</summary>
    Transpose,
    /// <summary>Bounded tensor slice.</summary>
    Slice,
    /// <summary>Tensor concatenation.</summary>
    Concat,
    /// <summary>Bounded gather.</summary>
    Gather,
    /// <summary>Bounded scatter.</summary>
    Scatter,
    /// <summary>Stable softmax.</summary>
    Softmax,
    /// <summary>Rotary position embedding.</summary>
    Rope,
    /// <summary>Explicit masked causal attention.</summary>
    CausalAttention,
    /// <summary>Reads one state slot.</summary>
    StateRead,
    /// <summary>Appends to one state slot.</summary>
    StateAppend,
    /// <summary>Commits staged state.</summary>
    StateCommit,
    /// <summary>Rolls staged state back.</summary>
    StateRollback,
    /// <summary>Selects bounded routes.</summary>
    TopKRoute,
    /// <summary>Structured conditional branch.</summary>
    Branch,
    /// <summary>Explicit branch merge.</summary>
    Merge,
    /// <summary>Structured bounded loop.</summary>
    Loop,
    /// <summary>Selects logits from a larger tensor.</summary>
    SelectLogits,
    /// <summary>Samples one token.</summary>
    Sample,
    /// <summary>Entry-point output.</summary>
    Output,
}

/// <summary>Merge semantics for explicit control-flow fan-in.</summary>
public enum MergeMode
{
    /// <summary>Concatenates active inputs.</summary>
    Concat,
    /// <summary>Adds active inputs.</summary>
    Add,
    /// <summary>Uses explicit gates to combine active inputs.</summary>
    GatedSum,
}

/// <summary>An SSA tensor value declared by the model graph.</summary>
/// <param name="Id">Stable value identity.</param>
/// <param name="Shape">Logical bounded shape.</param>
/// <param name="NumericType">Storage, compute, and accumulator types.</param>
public sealed record GraphValue(ValueId Id, TensorShape Shape, NumericType NumericType);

/// <summary>Schema for one mutable state slot.</summary>
/// <param name="Id">Stable state identity.</param>
/// <param name="Shape">Bounded per-entry shape.</param>
/// <param name="NumericType">Stored and computed numeric types.</param>
/// <param name="HasInitialValue">Whether a read is valid before the first graph writer.</param>
public sealed record StateSlotDescriptor(
    StateSlotId Id,
    TensorShape Shape,
    NumericType NumericType,
    bool HasInitialValue = false);

/// <summary>Structured loop bounds and loop-carried SSA values.</summary>
public sealed class LoopDescriptor
{
    /// <summary>Creates an immutable structured-loop contract.</summary>
    public LoopDescriptor(
        int maximumIterations,
        IEnumerable<ValueId> carriedInputs,
        IEnumerable<ValueId> carriedOutputs)
    {
        ArgumentNullException.ThrowIfNull(carriedInputs);
        ArgumentNullException.ThrowIfNull(carriedOutputs);
        MaximumIterations = maximumIterations;
        CarriedInputs = Copy(carriedInputs);
        CarriedOutputs = Copy(carriedOutputs);
    }

    /// <summary>Hard iteration limit.</summary>
    public int MaximumIterations { get; }
    /// <summary>Values entering one iteration.</summary>
    public IReadOnlyList<ValueId> CarriedInputs { get; }
    /// <summary>Values leaving one iteration.</summary>
    public IReadOnlyList<ValueId> CarriedOutputs { get; }

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T> source) =>
        Array.AsReadOnly([.. source]);
}

/// <summary>One typed operation and its data and effect dependencies.</summary>
/// <remarks>Creates one graph operation.</remarks>
public sealed class GraphNode(
    NodeId id,
    GraphOperationKind operation,
    IEnumerable<ValueId>? inputs = null,
    IEnumerable<ValueId>? outputs = null,
    IEnumerable<StateSlotId>? stateReads = null,
    IEnumerable<StateSlotId>? stateWrites = null,
    IEnumerable<EffectToken>? effectInputs = null,
    IEnumerable<EffectToken>? effectOutputs = null,
    MergeMode? mergeMode = null,
    LoopDescriptor? loop = null,
    TensorId? tensor = null,
    GraphOperationAttributes? attributes = null)
{

    /// <summary>Stable node identity.</summary>
    public NodeId Id { get; } = id;
    /// <summary>Operation semantics.</summary>
    public GraphOperationKind Operation { get; } = operation;
    /// <summary>Input SSA values.</summary>
    public IReadOnlyList<ValueId> Inputs { get; } = Copy(inputs);
    /// <summary>Output SSA values.</summary>
    public IReadOnlyList<ValueId> Outputs { get; } = Copy(outputs);
    /// <summary>State slots read by this operation.</summary>
    public IReadOnlyList<StateSlotId> StateReads { get; } = Copy(stateReads);
    /// <summary>State slots written by this operation.</summary>
    public IReadOnlyList<StateSlotId> StateWrites { get; } = Copy(stateWrites);
    /// <summary>Effect dependencies consumed by this operation.</summary>
    public IReadOnlyList<EffectToken> EffectInputs { get; } = Copy(effectInputs);
    /// <summary>Effect tokens produced by this operation.</summary>
    public IReadOnlyList<EffectToken> EffectOutputs { get; } = Copy(effectOutputs);
    /// <summary>Explicit merge semantics when this is a merge node.</summary>
    public MergeMode? MergeMode { get; } = mergeMode;
    /// <summary>Structured loop contract when this is a loop node.</summary>
    public LoopDescriptor? Loop { get; } = loop;
    /// <summary>Immutable tensor identity bound to a Constant node.</summary>
    public TensorId? Tensor { get; } = tensor;
    /// <summary>Typed parameters required by this operation.</summary>
    public GraphOperationAttributes? Attributes { get; } = attributes;

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T>? source) =>
        Array.AsReadOnly<T>(source is null ? [] : [.. source]);
}

/// <summary>Named callable graph boundary.</summary>
public sealed class GraphEntryPoint
{
    /// <summary>Creates an immutable callable graph boundary.</summary>
    public GraphEntryPoint(
        EntryPointId id,
        string name,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        Id = id;
        Name = name;
        Inputs = Copy(inputs);
        Outputs = Copy(outputs);
    }

    /// <summary>Stable entry-point identity.</summary>
    public EntryPointId Id { get; }
    /// <summary>Human-readable unique name.</summary>
    public string Name { get; }
    /// <summary>Values supplied by the caller.</summary>
    public IReadOnlyList<ValueId> Inputs { get; }
    /// <summary>Values returned to the caller.</summary>
    public IReadOnlyList<ValueId> Outputs { get; }

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T> source) =>
        Array.AsReadOnly([.. source]);
}

/// <summary>Portable typed model graph before backend lowering.</summary>
public sealed class ModelGraph
{
    /// <summary>Creates an immutable model graph snapshot.</summary>
    public ModelGraph(
        GraphVersion graphVersion,
        OpSetVersion opSetVersion,
        IEnumerable<GraphValue> values,
        IEnumerable<GraphNode> nodes,
        IEnumerable<StateSlotDescriptor>? stateSlots,
        IEnumerable<GraphEntryPoint> entryPoints,
        IEnumerable<RegionDescriptor>? regions = null)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(nodes);
        ArgumentNullException.ThrowIfNull(entryPoints);
        GraphVersion = graphVersion;
        OpSetVersion = opSetVersion;
        Values = Copy(values);
        Nodes = Copy(nodes);
        StateSlots = Copy(stateSlots);
        EntryPoints = Copy(entryPoints);
        Regions = Copy(regions);
    }

    /// <summary>Portable graph schema version.</summary>
    public GraphVersion GraphVersion { get; }
    /// <summary>Operation semantic version.</summary>
    public OpSetVersion OpSetVersion { get; }
    /// <summary>All declared SSA values.</summary>
    public IReadOnlyList<GraphValue> Values { get; }
    /// <summary>Operations in required producer-before-consumer order.</summary>
    public IReadOnlyList<GraphNode> Nodes { get; }
    /// <summary>Mutable state schema.</summary>
    public IReadOnlyList<StateSlotDescriptor> StateSlots { get; }
    /// <summary>Callable graph boundaries.</summary>
    public IReadOnlyList<GraphEntryPoint> EntryPoints { get; }
    /// <summary>Coarse, potentially conditional execution regions.</summary>
    public IReadOnlyList<RegionDescriptor> Regions { get; }

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T>? source) =>
        Array.AsReadOnly<T>(source is null ? [] : [.. source]);
}
