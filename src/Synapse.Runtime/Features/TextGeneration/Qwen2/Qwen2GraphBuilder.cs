using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class Qwen2GraphBuilder
{
    private static readonly NumericType Float = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);
    private static readonly NumericType Integer = new(
        StorageDataType.I32,
        ComputeDataType.I32,
        AccumulatorDataType.I32);

    public static ModelGraph Build(
        GgufFile file,
        int layerCount,
        int hiddenSize,
        int feedForwardSize,
        int keyValueWidth,
        int maximumContextSize,
        RopeScaling? ropeScaling)
    {
        var context = new Qwen2GraphBuildContext();
        var hidden = AddInputRegion(context, file, hiddenSize, out var token);
        var position = context.AddInput(Vector(1), Integer);
        var attentionHeads = file.GetRequiredInt32("qwen2.attention.head_count");
        var keyValueHeads = file.GetRequiredInt32("qwen2.attention.head_count_kv");
        if (attentionHeads <= 0 || keyValueHeads <= 0 ||
            attentionHeads % keyValueHeads != 0 || hiddenSize % attentionHeads != 0)
        {
            throw new InvalidDataException("Qwen2 attention-head metadata is incompatible with its hidden size.");
        }

        var headDimension = hiddenSize / attentionHeads;
        var rope = new RopeAttributes(
            file.GetRequiredSingle("qwen2.rope.freq_base"),
            headDimension,
            RotaryLayout.NeoX,
            ropeScaling);
        var normalizationEpsilon = file.GetRequiredSingle("qwen2.attention.layer_norm_rms_epsilon");

        for (var layer = 0; layer < layerCount; layer++)
        {
            hidden = Qwen2LayerGraphBuilder.AddTransformerRegion(
                context,
                file,
                layer,
                hidden,
                position,
                hiddenSize,
                feedForwardSize,
                keyValueWidth,
                maximumContextSize,
                attentionHeads,
                keyValueHeads,
                headDimension,
                rope,
                normalizationEpsilon);
        }

        var logits = AddOutputRegion(context, file, hidden, normalizationEpsilon);
        var entryPoint = new GraphEntryPoint(
            new EntryPointId(1),
            "forward-token",
            [token.Id, position.Id],
            [logits.Id]);
        return new ModelGraph(
            new GraphVersion(1, 0),
            new OpSetVersion(1, 0),
            context.Values,
            context.Nodes,
            context.StateSlots,
            [entryPoint],
            context.Regions,
            context.Weights);
    }

    private static GraphValue AddInputRegion(
        Qwen2GraphBuildContext context,
        GgufFile file,
        int hiddenSize,
        out GraphValue token)
    {
        var nodes = new List<NodeId>();
        var weights = new List<TensorId>();
        token = context.AddInput(Vector(1), Integer);
        var embedding = context.AddWeight(
            file.SourceFile,
            file.GetRequiredTensor("token_embd.weight"),
            nodes,
            weights);
        var hidden = context.Emit(
            GraphOperationKind.Embedding,
            [token.Id, embedding.Value.Id],
            Vector(hiddenSize),
            Float,
            nodes);
        context.AddRegion(
            nodes,
            [token.Id],
            [hidden.Id],
            weights,
            stateReads: null,
            stateWrites: null,
            "Qwen2",
            "Embedding");
        return hidden;
    }

    private static GraphValue AddOutputRegion(
        Qwen2GraphBuildContext context,
        GgufFile file,
        GraphValue hidden,
        float normalizationEpsilon)
    {
        var nodes = new List<NodeId>();
        var weights = new List<TensorId>();
        var normWeight = AddWeight(context, file, "output_norm.weight", nodes, weights);
        var normalized = EmitNormalization(context, hidden, normWeight, normalizationEpsilon, nodes);
        var outputWeight = context.AddWeight(
            file.SourceFile,
            file.GetRequiredTensor("output.weight"),
            nodes,
            weights);
        var vocabularySize = outputWeight.Value.Shape.Dimensions[0].Maximum;
        var logits = context.Emit(
            GraphOperationKind.QuantizedLinear,
            [normalized.Id, outputWeight.Value.Id],
            Vector(vocabularySize),
            Float,
            nodes);
        _ = context.AddOutput(logits.Id);
        context.AddRegion(
            nodes,
            [hidden.Id],
            [logits.Id],
            weights,
            stateReads: null,
            stateWrites: null,
            "Qwen2",
            "Logits");
        return logits;
    }

    private static GraphValue AddWeight(
        Qwen2GraphBuildContext context,
        GgufFile file,
        string name,
        ICollection<NodeId> nodes,
        ICollection<TensorId> weights) => context.AddWeight(
            file.SourceFile,
            file.GetRequiredTensor(name),
            nodes,
            weights).Value;

    private static GraphValue EmitNormalization(
        Qwen2GraphBuildContext context,
        GraphValue input,
        GraphValue parameter,
        float epsilon,
        ICollection<NodeId> nodes) => context.Emit(
            GraphOperationKind.RmsNorm,
            [input.Id, parameter.Id],
            input.Shape,
            Float,
            nodes,
            attributes: new NormalizationAttributes(epsilon));

    private static TensorShape Vector(long size) => new(ShapeDimension.Fixed(size));

}
