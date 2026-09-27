using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

/// <summary>Stable model-graph verification failure categories.</summary>
public enum GraphDiagnosticCode
{
    /// <summary>The graph or operation-set major version is unsupported.</summary>
    UnsupportedVersion,
    /// <summary>An identity is declared more than once.</summary>
    DuplicateId,
    /// <summary>A referenced value, state slot, or effect token is absent.</summary>
    UnknownReference,
    /// <summary>An SSA value or effect token has multiple producers.</summary>
    MultipleProducers,
    /// <summary>A consumed value has no producer.</summary>
    MissingProducer,
    /// <summary>A producer appears after its consumer.</summary>
    ProducerAfterConsumer,
    /// <summary>The dependency graph contains an unstructured cycle.</summary>
    IllegalCycle,
    /// <summary>A tensor shape is invalid or incompatible.</summary>
    ShapeMismatch,
    /// <summary>A numeric type combination is invalid for an operation.</summary>
    NumericTypeMismatch,
    /// <summary>A state read has no initial value or preceding writer.</summary>
    UnwrittenStateRead,
    /// <summary>Writers of one state slot have no serialization dependency.</summary>
    UnorderedStateWriters,
    /// <summary>A structured loop contract is missing or invalid.</summary>
    InvalidLoop,
    /// <summary>An entry point is missing or references invalid values.</summary>
    InvalidEntryPoint,
    /// <summary>An execution region is empty, overlapping, or otherwise malformed.</summary>
    InvalidRegion,
    /// <summary>A declared region boundary differs from its member-node dependencies.</summary>
    RegionBoundaryMismatch,
    /// <summary>A constant tensor binding is missing, duplicated, or attached to another operation.</summary>
    InvalidTensorBinding,
    /// <summary>An operation is missing required typed attributes or declares incompatible attributes.</summary>
    InvalidOperationAttributes,
    /// <summary>An operation kind or contract is not supported by this verifier.</summary>
    UnsupportedOperation,
}

/// <summary>One structured verifier diagnostic.</summary>
/// <param name="Code">Stable failure category.</param>
/// <param name="Message">Actionable explanation.</param>
/// <param name="NodeId">Related node when applicable.</param>
public sealed record GraphDiagnostic(
    GraphDiagnosticCode Code,
    string Message,
    NodeId? NodeId = null);

/// <summary>Immutable graph verification outcome.</summary>
public sealed class GraphVerificationResult
{
    internal GraphVerificationResult(IEnumerable<GraphDiagnostic> diagnostics)
    {
        Diagnostics = Array.AsReadOnly([.. diagnostics]);
    }

    /// <summary>Whether no verifier errors were found.</summary>
    public bool IsValid => Diagnostics.Count == 0;

    /// <summary>All deterministic diagnostics in discovery order.</summary>
    public IReadOnlyList<GraphDiagnostic> Diagnostics { get; }
}
