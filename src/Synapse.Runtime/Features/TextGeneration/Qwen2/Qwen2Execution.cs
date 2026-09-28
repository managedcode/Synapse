namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Qwen2 hyperparameters read from GGUF metadata plus the session context bound.</summary>
internal sealed record Qwen2Dimensions(
    int LayerCount,
    int HiddenSize,
    int FeedForwardSize,
    int AttentionHeads,
    int KeyValueHeads,
    int VocabularySize,
    int ContextSize,
    float RopeTheta,
    float RmsNormEpsilon)
{
    public int HeadDimension => HiddenSize / AttentionHeads;

    public int KvWidth => KeyValueHeads * HeadDimension;
}

/// <summary>
/// One backend's direct single-session evaluation; positions always start at zero for a prompt. Callers hold
/// the model's execution gate.
/// </summary>
internal interface IQwen2Executor : IDisposable
{
    string RuntimeProfile { get; }

    string KernelImplementation { get; }

    /// <summary>Evaluates prompt positions <c>0..n-1</c> and returns the final position's logits.</summary>
    ReadOnlyMemory<float> Prefill(IReadOnlyList<int> tokens);

    /// <summary>Evaluates one token at <paramref name="position"/> and returns its logits.</summary>
    ReadOnlyMemory<float> Decode(int token, int position);
}
