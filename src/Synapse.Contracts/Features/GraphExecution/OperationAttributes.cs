namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Typed, operation-specific parameters required to execute one graph node.</summary>
public abstract record GraphOperationAttributes;

/// <summary>Parameters for RMS or layer normalization.</summary>
/// <param name="Epsilon">Positive numerical-stability constant.</param>
public sealed record NormalizationAttributes(float Epsilon) : GraphOperationAttributes;

/// <summary>Supported rotary-coordinate layouts.</summary>
public enum RotaryLayout
{
    /// <summary>Pairs the first and second halves of each attention head.</summary>
    NeoX,
    /// <summary>Pairs adjacent coordinates in each attention head.</summary>
    Interleaved,
}

/// <summary>Parameters for rotary position embedding.</summary>
/// <param name="Theta">Positive rotary frequency base.</param>
/// <param name="HeadDimension">Coordinates in one attention head.</param>
/// <param name="Layout">Coordinate-pairing layout.</param>
public sealed record RopeAttributes(
    float Theta,
    int HeadDimension,
    RotaryLayout Layout) : GraphOperationAttributes;

/// <summary>Attention-mask semantics.</summary>
public enum AttentionMaskKind
{
    /// <summary>Each position observes only itself and earlier positions.</summary>
    Causal,
}

/// <summary>Parameters for grouped-query causal attention.</summary>
/// <param name="QueryHeads">Number of query heads.</param>
/// <param name="KeyValueHeads">Number of key/value heads.</param>
/// <param name="HeadDimension">Coordinates in one head.</param>
/// <param name="Scale">Multiplier applied to query-key scores.</param>
/// <param name="Mask">Explicit attention-mask semantics.</param>
/// <param name="HandlesPositionHoles">Whether the kernel masks missing state positions.</param>
public sealed record CausalAttentionAttributes(
    int QueryHeads,
    int KeyValueHeads,
    int HeadDimension,
    float Scale,
    AttentionMaskKind Mask,
    bool HandlesPositionHoles = false) : GraphOperationAttributes;

/// <summary>Axis over which a top-k routing decision is computed.</summary>
public enum TopKRouteAxis
{
    /// <summary>Routes independently from the current feature vector.</summary>
    Feature,
    /// <summary>Routes after observing values across the sequence.</summary>
    Sequence,
}

/// <summary>Deterministic tie-breaking policy for top-k routing.</summary>
public enum RouteTiePolicy
{
    /// <summary>Equal scores prefer the lowest source index.</summary>
    StableLowestIndex,
}

/// <summary>Parameters for an explicit top-k routing operation.</summary>
/// <param name="K">Number of selected slots.</param>
/// <param name="Axis">Axis used to compute the route.</param>
/// <param name="TiePolicy">Deterministic score tie behavior.</param>
/// <param name="Capacity">Optional maximum assignments per route.</param>
public sealed record TopKRouteAttributes(
    int K,
    TopKRouteAxis Axis,
    RouteTiePolicy TiePolicy,
    int? Capacity) : GraphOperationAttributes;
