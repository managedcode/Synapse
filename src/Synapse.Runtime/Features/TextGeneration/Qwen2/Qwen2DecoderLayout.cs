using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// The Qwen2 family adapter's only GPU-facing job: map GGUF tensor names onto the brand-neutral
/// <see cref="DenseDecoderLayout"/>. Qwen2 has Q/K/V biases, NeoX RoPE, and an untied output projection.
/// </summary>
internal static class Qwen2DecoderLayout
{
    public static DenseDecoderLayout Describe(GgufFile file, DecoderDimensions dimensions)
    {
        var layers = new DenseDecoderLayer[dimensions.LayerCount];
        for (var layer = 0; layer < layers.Length; layer++)
        {
            var prefix = $"blk.{layer}";
            layers[layer] = new DenseDecoderLayer(
                Weight(file, $"{prefix}.attn_norm.weight"),
                Weight(file, $"{prefix}.attn_q.weight"),
                Weight(file, $"{prefix}.attn_k.weight"),
                Weight(file, $"{prefix}.attn_v.weight"),
                Weight(file, $"{prefix}.attn_q.bias"),
                Weight(file, $"{prefix}.attn_k.bias"),
                Weight(file, $"{prefix}.attn_v.bias"),
                Weight(file, $"{prefix}.attn_output.weight"),
                Weight(file, $"{prefix}.ffn_norm.weight"),
                Weight(file, $"{prefix}.ffn_gate.weight"),
                Weight(file, $"{prefix}.ffn_up.weight"),
                Weight(file, $"{prefix}.ffn_down.weight"));
        }

        return new DenseDecoderLayout(
            "qwen2",
            dimensions,
            RotaryLayout.NeoX,
            Weight(file, "token_embd.weight"),
            Weight(file, "output_norm.weight"),
            Weight(file, "output.weight"),
            layers);
    }

    private static DecoderWeight Weight(GgufFile file, string name)
    {
        var tensor = file.GetRequiredTensor(name);
        return new DecoderWeight(
            new WeightSourceRange(file.SourceFile, tensor.Offset, tensor.ByteLength),
            Qwen2GraphBuildContext.ToEncoding(tensor));
    }
}
