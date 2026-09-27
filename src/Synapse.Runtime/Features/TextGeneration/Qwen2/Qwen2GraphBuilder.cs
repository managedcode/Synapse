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
        int contextSize)
    {
        var context = new Qwen2GraphBuildContext();
        var hidden = AddInputRegion(context, file, hiddenSize, out var token);

        for (var layer = 0; layer < layerCount; layer++)
        {
            hidden = Qwen2LayerGraphBuilder.AddTransformerRegion(
                context,
                file,
                layer,
                hidden,
                hiddenSize,
                feedForwardSize,
                keyValueWidth,
                contextSize);
        }

        var logits = AddOutputRegion(context, file, hidden);
        var entryPoint = new GraphEntryPoint(new EntryPointId(1), "forward-token", [token.Id], [logits.Id]);
        return new ModelGraph(
            new GraphVersion(1, 0),
            new OpSetVersion(1, 0),
            context.Values,
            context.Nodes,
            context.StateSlots,
            [entryPoint],
            context.Regions);
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
        var embedding = context.AddWeight(file.GetRequiredTensor("token_embd.weight"), nodes, weights);
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
        GraphValue hidden)
    {
        var nodes = new List<NodeId>();
        var weights = new List<TensorId>();
        var normWeight = AddWeight(context, file, "output_norm.weight", nodes, weights);
        var normalized = EmitUnary(context, GraphOperationKind.RmsNorm, hidden, normWeight, nodes);
        var outputWeight = context.AddWeight(file.GetRequiredTensor("output.weight"), nodes, weights);
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
        ICollection<TensorId> weights) => context.AddWeight(file.GetRequiredTensor(name), nodes, weights).Value;

    private static GraphValue EmitUnary(
        Qwen2GraphBuildContext context,
        GraphOperationKind operation,
        GraphValue input,
        GraphValue parameter,
        ICollection<NodeId> nodes) => context.Emit(
            operation,
            [input.Id, parameter.Id],
            input.Shape,
            Float,
            nodes);

    private static TensorShape Vector(long size) => new(ShapeDimension.Fixed(size));

}
