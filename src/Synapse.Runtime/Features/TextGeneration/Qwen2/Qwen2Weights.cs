using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed record Qwen2LayerWeights(
    float[] AttentionNorm,
    GgufTensorInfo Query,
    GgufTensorInfo Key,
    GgufTensorInfo Value,
    float[] QueryBias,
    float[] KeyBias,
    float[] ValueBias,
    GgufTensorInfo AttentionOutput,
    float[] FeedForwardNorm,
    GgufTensorInfo FeedForwardGate,
    GgufTensorInfo FeedForwardUp,
    GgufTensorInfo FeedForwardDown);

internal sealed record Qwen2Weights(
    GgufTensorInfo TokenEmbedding,
    IReadOnlyList<Qwen2LayerWeights> Layers,
    float[] OutputNorm,
    GgufTensorInfo Output);
