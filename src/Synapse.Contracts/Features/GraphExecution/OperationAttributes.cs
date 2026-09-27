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
public sealed record CausalAttentionAttributes(
    int QueryHeads,
    int KeyValueHeads,
    int HeadDimension,
    float Scale,
    AttentionMaskKind Mask) : GraphOperationAttributes;
