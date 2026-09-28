namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>Element type of the KV cache; FP16 is an explicit GPU numerical profile (ADR-012).</summary>
public enum KvCachePrecision
{
    /// <summary>FP32 keys and values; the default and the only CPU layout.</summary>
    Fp32,

    /// <summary>FP16 keys and values with FP32 accumulation; half the KV memory and bandwidth.</summary>
    Fp16,
}
