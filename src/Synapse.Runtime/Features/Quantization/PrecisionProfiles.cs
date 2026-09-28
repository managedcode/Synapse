using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>One encoding a tensor may use, with its stored size and measured distortion.</summary>
/// <param name="EncodingId">Stable encoding identity; the first option is the source encoding.</param>
/// <param name="Bytes">Exact stored bytes of the tensor in this encoding.</param>
/// <param name="Distortion">Measured relative output distortion against the source.</param>
public sealed record PrecisionOption(string EncodingId, long Bytes, double Distortion);

/// <summary>A tensor and its encodings ordered from source precision to the lowest.</summary>
/// <param name="Tensor">Graph tensor identity.</param>
/// <param name="Options">Encodings with strictly decreasing stored size.</param>
/// <param name="PinnedHigh">Whether the tensor must keep its first supported encoding.</param>
public sealed record PrecisionCandidate(
    TensorId Tensor,
    IReadOnlyList<PrecisionOption> Options,
    bool PinnedHigh = false);

/// <summary>Constraints for one precision selection, typically one device's weight budget.</summary>
/// <param name="BudgetBytes">Maximum stored weight bytes.</param>
/// <param name="AllowApproximation">Whether the session opted into a quality-bounded profile.</param>
/// <param name="SupportedEncodings">Encodings with kernels on the target device, or all when null.</param>
public sealed record PrecisionPolicy(
    long BudgetBytes,
    bool AllowApproximation,
    IReadOnlySet<string>? SupportedEncodings = null);

/// <summary>The encoding chosen for one tensor.</summary>
/// <param name="Tensor">Graph tensor identity.</param>
/// <param name="EncodingId">Chosen encoding.</param>
/// <param name="Bytes">Stored bytes in the chosen encoding.</param>
/// <param name="Distortion">Measured distortion of the chosen encoding.</param>
/// <param name="IsDemoted">Whether the chosen encoding differs from the source encoding.</param>
public sealed record PrecisionAssignment(
    TensorId Tensor,
    string EncodingId,
    long Bytes,
    double Distortion,
    bool IsDemoted);

/// <summary>An effective-weights profile: which encoding every tensor uses.</summary>
/// <param name="Assignments">Assignments ordered by tensor identity.</param>
/// <param name="TotalBytes">Stored weight bytes of the profile.</param>
/// <param name="TotalDistortion">Sum of per-tensor distortion estimates.</param>
/// <param name="IsApproximate">Whether any tensor is demoted, making the profile quality-bounded.</param>
/// <param name="ProfileHash">Identity of the effective weights, used in plan and KV cache keys.</param>
public sealed record PrecisionPlan(
    IReadOnlyList<PrecisionAssignment> Assignments,
    long TotalBytes,
    double TotalDistortion,
    bool IsApproximate,
    ContentHash ProfileHash);

/// <summary>A change of one tensor's encoding between two profiles.</summary>
/// <param name="Tensor">Graph tensor identity.</param>
/// <param name="FromEncoding">Encoding in the current profile.</param>
/// <param name="ToEncoding">Encoding in the target profile.</param>
/// <param name="BytesDelta">Stored-size change; positive means more precision.</param>
public sealed record PrecisionChange(TensorId Tensor, string FromEncoding, string ToEncoding, long BytesDelta)
{
    /// <summary>Whether the change restores precision.</summary>
    public bool IsPromotion => BytesDelta > 0;
}

/// <summary>Stable categories of precision-selection failures.</summary>
public enum PrecisionBudgetFailure
{
    /// <summary>Even the lowest allowed encodings exceed the budget.</summary>
    InsufficientBudget,
    /// <summary>The budget requires demotion, but the policy forbids approximation.</summary>
    ApproximationNotAllowed,
    /// <summary>A tensor has no encoding supported by the target device.</summary>
    UnsupportedEncoding,
}

/// <summary>A typed precision-selection failure with the smallest feasible size.</summary>
/// <remarks>Creates a failure with its category and minimum feasible bytes.</remarks>
public sealed class PrecisionBudgetException(PrecisionBudgetFailure failure, long minimumBytes, string message) : InvalidOperationException(message)
{

    /// <summary>Stable failure category.</summary>
    public PrecisionBudgetFailure Failure { get; } = failure;

    /// <summary>Smallest stored size achievable under the failing policy.</summary>
    public long MinimumBytes { get; } = minimumBytes;
}
