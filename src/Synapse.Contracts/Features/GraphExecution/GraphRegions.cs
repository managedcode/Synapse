namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>A coarse executable subgraph used for activation-wave scheduling.</summary>
public sealed class RegionDescriptor
{
    /// <summary>Creates one region with explicit activation and skip semantics.</summary>
    public RegionDescriptor(
        RegionId id,
        IEnumerable<NodeId> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        IEnumerable<TensorId>? requiredWeights,
        IEnumerable<StateSlotId>? stateReads,
        IEnumerable<StateSlotId>? stateWrites,
        RegionActivation activation,
        IEnumerable<string>? semanticAnnotations = null)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        ArgumentNullException.ThrowIfNull(activation);
        Id = id;
        Nodes = Copy(nodes);
        Inputs = Copy(inputs);
        Outputs = Copy(outputs);
        RequiredWeights = Copy(requiredWeights);
        StateReads = Copy(stateReads);
        StateWrites = Copy(stateWrites);
        Activation = activation;
        SemanticAnnotations = Copy(semanticAnnotations);
    }

    /// <summary>Stable region identity.</summary>
    public RegionId Id { get; }
    /// <summary>Operations scheduled as this coarse unit.</summary>
    public IReadOnlyList<NodeId> Nodes { get; }
    /// <summary>Values entering from another region or entry point.</summary>
    public IReadOnlyList<ValueId> Inputs { get; }
    /// <summary>Values consumed outside this region.</summary>
    public IReadOnlyList<ValueId> Outputs { get; }
    /// <summary>Immutable weights required before activation.</summary>
    public IReadOnlyList<TensorId> RequiredWeights { get; }
    /// <summary>State slots observed by the region.</summary>
    public IReadOnlyList<StateSlotId> StateReads { get; }
    /// <summary>State slots mutated by the region.</summary>
    public IReadOnlyList<StateSlotId> StateWrites { get; }
    /// <summary>Formal decision, provenance, and skip contract.</summary>
    public RegionActivation Activation { get; }
    /// <summary>Non-semantic labels such as CSharp or Reasoning.</summary>
    public IReadOnlyList<string> SemanticAnnotations { get; }

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T>? source) =>
        Array.AsReadOnly<T>(source is null ? [] : [.. source]);
}
