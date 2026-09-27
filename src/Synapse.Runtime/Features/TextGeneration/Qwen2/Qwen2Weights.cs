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

internal static class Qwen2WeightLoader
{
    public static Qwen2Weights Load(GgufFile file, int layerCount)
    {
        var layers = new Qwen2LayerWeights[layerCount];
        for (var layer = 0; layer < layerCount; layer++)
        {
            var prefix = $"blk.{layer}";
            layers[layer] = new Qwen2LayerWeights(
                file.ReadFloat32Tensor($"{prefix}.attn_norm.weight"),
                file.GetRequiredTensor($"{prefix}.attn_q.weight"),
                file.GetRequiredTensor($"{prefix}.attn_k.weight"),
                file.GetRequiredTensor($"{prefix}.attn_v.weight"),
                file.ReadFloat32Tensor($"{prefix}.attn_q.bias"),
                file.ReadFloat32Tensor($"{prefix}.attn_k.bias"),
                file.ReadFloat32Tensor($"{prefix}.attn_v.bias"),
                file.GetRequiredTensor($"{prefix}.attn_output.weight"),
                file.ReadFloat32Tensor($"{prefix}.ffn_norm.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_gate.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_up.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_down.weight"));
        }

        return new Qwen2Weights(
            file.GetRequiredTensor("token_embd.weight"),
            layers,
            file.ReadFloat32Tensor("output_norm.weight"),
            file.GetRequiredTensor("output.weight"));
    }
}
