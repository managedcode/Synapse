using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.Runtime.Features.ExecutionPlanning;

/// <summary>What the placement search minimizes.</summary>
public enum PlacementObjective
{
    /// <summary>One sequence's decode step: stage and link times add up.</summary>
    Latency,
    /// <summary>Many concurrent sequences in a pipeline: the slowest stage or link bounds the rate.</summary>
    Throughput,
}

/// <summary>One layer region: its tensors with precision options and its KV growth.</summary>
/// <param name="Index">Zero-based position in execution order.</param>
/// <param name="Tensors">Weight tensors of the region with measured precision options.</param>
/// <param name="KvBytesPerToken">KV bytes this region adds per cached token.</param>
public sealed record PlacementLayer(int Index, IReadOnlyList<PrecisionCandidate> Tensors, long KvBytesPerToken);

/// <summary>A measured device or worker profile.</summary>
/// <param name="Id">Stable device or worker identity.</param>
/// <param name="MemoryBudgetBytes">Bytes available for weights and KV reservations.</param>
/// <param name="WeightBytesPerSecond">Measured effective weight-streaming rate for decode.</param>
/// <param name="LayerOverheadSeconds">Measured fixed dispatch cost per region.</param>
/// <param name="SupportedEncodings">Encodings with kernels on this device, or all when null.</param>
public sealed record DeviceProfile(
    string Id,
    long MemoryBudgetBytes,
    double WeightBytesPerSecond,
    double LayerOverheadSeconds,
    IReadOnlySet<string>? SupportedEncodings = null);

/// <summary>A measured link between consecutive pipeline stages.</summary>
/// <param name="LatencySeconds">Per-message latency.</param>
/// <param name="BytesPerSecond">Sustained transfer rate.</param>
public sealed record LinkProfile(double LatencySeconds, double BytesPerSecond);

/// <summary>Inputs to one bounded placement search.</summary>
/// <param name="Layers">Layer regions in execution order.</param>
/// <param name="Devices">One to four candidate devices.</param>
/// <param name="Link">Link profile used between any two stages.</param>
/// <param name="ActivationBytesPerToken">Bytes crossing a stage boundary per decoded token.</param>
/// <param name="ContextTokens">Tokens of KV reserved per placed session.</param>
/// <param name="AllowApproximation">Whether the session opted into quality-bounded precision.</param>
/// <param name="Objective">What the search minimizes.</param>
public sealed record PlacementRequest(
    IReadOnlyList<PlacementLayer> Layers,
    IReadOnlyList<DeviceProfile> Devices,
    LinkProfile Link,
    long ActivationBytesPerToken,
    int ContextTokens,
    bool AllowApproximation,
    PlacementObjective Objective);

/// <summary>One contiguous range of layer regions on one device.</summary>
/// <param name="DeviceId">Device hosting the range.</param>
/// <param name="FirstLayer">First layer index, inclusive.</param>
/// <param name="LastLayer">Last layer index, inclusive.</param>
/// <param name="Precision">Importance-aware precision chosen for the range's tensors.</param>
/// <param name="ReservedBytes">Weights plus KV reservation.</param>
/// <param name="MemoryBudgetBytes">Device budget the reservation must respect.</param>
/// <param name="EstimatedSeconds">Estimated decode time of the range per token.</param>
public sealed record PlacementStage(
    string DeviceId,
    int FirstLayer,
    int LastLayer,
    PrecisionPlan Precision,
    long ReservedBytes,
    long MemoryBudgetBytes,
    double EstimatedSeconds);

/// <summary>A chosen pipeline of stages with its estimates.</summary>
/// <param name="Stages">Stages in execution order.</param>
/// <param name="EstimatedTokenSeconds">Estimated seconds per token under the objective.</param>
/// <param name="TotalDistortion">Sum of per-tensor distortion estimates.</param>
/// <param name="IsApproximate">Whether any stage demotes a tensor.</param>
/// <param name="Objective">Objective the plan was optimized for.</param>
public sealed record PlacementPlan(
    IReadOnlyList<PlacementStage> Stages,
    double EstimatedTokenSeconds,
    double TotalDistortion,
    bool IsApproximate,
    PlacementObjective Objective);

/// <summary>Stable categories of placement failures.</summary>
public enum PlacementFailure
{
    /// <summary>No partition fits the devices, even with the lowest allowed precision.</summary>
    InsufficientMemory,
    /// <summary>A partition exists only with approximation, which the request forbids.</summary>
    ApproximationNotAllowed,
}

/// <summary>A typed placement failure raised before any allocation.</summary>
/// <remarks>Creates a failure with its stable category.</remarks>
public sealed class PlacementException(PlacementFailure failure, string message) : InvalidOperationException(message)
{

    /// <summary>Stable failure category.</summary>
    public PlacementFailure Failure { get; } = failure;
}
