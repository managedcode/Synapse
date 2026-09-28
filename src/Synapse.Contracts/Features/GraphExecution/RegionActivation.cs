namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Complete execution and skip contract for one region.</summary>
/// <param name="Decision">How the scheduler decides whether the region executes.</param>
/// <param name="Provenance">Why skipping is a legitimate graph transformation.</param>
/// <param name="Skip">What the region produces when it does not execute.</param>
public sealed record RegionActivation(
    ActivationDecision Decision,
    EligibilityProvenance Provenance,
    SkipSemantics Skip);

/// <summary>Scheduler-visible activation decision.</summary>
public abstract record ActivationDecision;

/// <summary>The region executes for every invocation.</summary>
public sealed record AlwaysActive : ActivationDecision;

/// <summary>A graph-produced boolean controls region activation.</summary>
public sealed record PredicateDecision(ValueId Predicate, RouteScope Scope) : ActivationDecision;

/// <summary>One slot from a graph-produced top-k route controls activation.</summary>
public sealed record RouteSlotDecision(ValueId Route, int Slot, RouteScope Scope) : ActivationDecision;

/// <summary>A named admission profile controls activation.</summary>
public sealed record ProfileDecision(string ProfileKey) : ActivationDecision;

/// <summary>Lifetime over which an activation decision is evaluated.</summary>
public enum RouteScope
{
    /// <summary>Once for the whole session.</summary>
    Session,
    /// <summary>Once for one decode or prefill step.</summary>
    Step,
    /// <summary>Independently for tokens in the current batch.</summary>
    TokenInBatch,
}

/// <summary>Evidence that permits the declared skip behavior.</summary>
public abstract record EligibilityProvenance;

/// <summary>Skipping is part of the model's exact graph structure.</summary>
public sealed record StructuralProvenance : EligibilityProvenance;

/// <summary>Skipping is controlled by an explicit graph-author predicate.</summary>
public sealed record ProgrammedProvenance : EligibilityProvenance;

/// <summary>Skipping is selected by an immutable trained policy.</summary>
public sealed record TrainedPolicyProvenance(ContentHash PolicyHash) : EligibilityProvenance;

/// <summary>Skipping is backed by immutable quality evaluation evidence.</summary>
public sealed record ApproximateProvenance(ContentHash EvaluationHash) : EligibilityProvenance;

/// <summary>Observable output behavior when a region does not execute.</summary>
public abstract record SkipSemantics;

/// <summary>The region cannot be skipped.</summary>
public sealed record NotSkippable : SkipSemantics;

/// <summary>The region contributes no output values when skipped.</summary>
public sealed record OutputsAbsent : SkipSemantics;

/// <summary>Region outputs are replaced by declared region inputs when skipped.</summary>
public sealed record BypassOutputs : SkipSemantics
{
    /// <summary>Creates an immutable output-to-input bypass map.</summary>
    public BypassOutputs(IEnumerable<ValueBypass> map)
    {
        ArgumentNullException.ThrowIfNull(map);
        Map = Array.AsReadOnly<ValueBypass>([.. map]);
    }

    /// <summary>One mapping for every declared region output.</summary>
    public IReadOnlyList<ValueBypass> Map { get; }
}

/// <summary>One skipped output and the region input that replaces it.</summary>
public readonly record struct ValueBypass(ValueId Output, ValueId Input);
