namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Formal rule controlling whether an executable region may be skipped.</summary>
public abstract record ExecutionEligibility;

/// <summary>The region is required for every execution of its entry point.</summary>
public sealed record AlwaysRequiredEligibility : ExecutionEligibility;

/// <summary>The region executes when a graph-produced boolean predicate is true.</summary>
/// <param name="Predicate">Boolean SSA value controlling the region.</param>
public sealed record GraphPredicateEligibility(ValueId Predicate) : ExecutionEligibility;

/// <summary>The region is selected by a versioned learned routing policy.</summary>
/// <param name="PolicyHash">Identity of the immutable trained policy.</param>
public sealed record TrainedRouteEligibility(ContentHash PolicyHash) : ExecutionEligibility;

/// <summary>The region is optional only within an evaluated approximation profile.</summary>
/// <param name="EvaluationHash">Identity of the quality evidence.</param>
public sealed record ApproximateProfileEligibility(ContentHash EvaluationHash) : ExecutionEligibility;

/// <summary>A coarse executable subgraph used for activation-wave scheduling.</summary>
public sealed class RegionDescriptor
{
    /// <summary>Creates one region with explicit execution eligibility.</summary>
    public RegionDescriptor(
        RegionId id,
        IEnumerable<NodeId> nodes,
        IEnumerable<ValueId> inputs,
        IEnumerable<ValueId> outputs,
        IEnumerable<TensorId>? requiredWeights,
        IEnumerable<StateSlotId>? stateReads,
        IEnumerable<StateSlotId>? stateWrites,
        ExecutionEligibility eligibility,
        IEnumerable<string>? semanticAnnotations = null)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        ArgumentNullException.ThrowIfNull(eligibility);
        Id = id;
        Nodes = Copy(nodes);
        Inputs = Copy(inputs);
        Outputs = Copy(outputs);
        RequiredWeights = Copy(requiredWeights);
        StateReads = Copy(stateReads);
        StateWrites = Copy(stateWrites);
        Eligibility = eligibility;
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
    /// <summary>Formal execution or skip rule.</summary>
    public ExecutionEligibility Eligibility { get; }
    /// <summary>Non-semantic labels such as CSharp or Reasoning.</summary>
    public IReadOnlyList<string> SemanticAnnotations { get; }

    private static System.Collections.ObjectModel.ReadOnlyCollection<T> Copy<T>(IEnumerable<T>? source) =>
        Array.AsReadOnly<T>(source is null ? [] : [.. source]);
}
