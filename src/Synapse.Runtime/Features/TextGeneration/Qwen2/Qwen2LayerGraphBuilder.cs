using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class Qwen2LayerGraphBuilder
{
    private static readonly NumericType Float = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);

    public static GraphValue AddTransformerRegion(
        Qwen2GraphBuildContext context,
        GgufFile file,
        int layer,
        GraphValue hidden,
        GraphValue position,
        int hiddenSize,
        int feedForwardSize,
        int keyValueWidth,
        int maximumContextSize,
        int attentionHeads,
        int keyValueHeads,
        int headDimension,
        float ropeTheta,
        float normalizationEpsilon)
    {
        var nodes = new List<NodeId>();
        var weights = new List<TensorId>();
        var prefix = $"blk.{layer}";
        var attentionResidual = AddAttention(
            context,
            file,
            prefix,
            hidden,
            position,
            hiddenSize,
            keyValueWidth,
            maximumContextSize,
            attentionHeads,
            keyValueHeads,
            headDimension,
            ropeTheta,
            normalizationEpsilon,
            nodes,
            weights,
            out var keyState,
            out var valueState);
        var output = AddFeedForward(
            context,
            file,
            prefix,
            attentionResidual,
            hiddenSize,
            feedForwardSize,
            normalizationEpsilon,
            nodes,
            weights);
        context.AddRegion(
            nodes,
            [hidden.Id, position.Id],
            [output.Id],
            weights,
            [keyState, valueState],
            [keyState, valueState],
            "Qwen2",
            "DenseTransformerBlock",
            $"Layer:{layer}");
        return output;
    }

    private static GraphValue AddAttention(
        Qwen2GraphBuildContext context,
        GgufFile file,
        string prefix,
        GraphValue hidden,
        GraphValue position,
        int hiddenSize,
        int keyValueWidth,
        int maximumContextSize,
        int attentionHeads,
        int keyValueHeads,
        int headDimension,
        float ropeTheta,
        float normalizationEpsilon,
        ICollection<NodeId> nodes,
        ICollection<TensorId> weights,
        out StateSlotId keyState,
        out StateSlotId valueState)
    {
        var normWeight = AddWeight(context, file, $"{prefix}.attn_norm.weight", nodes, weights);
        var normalized = EmitNormalization(context, hidden, normWeight, normalizationEpsilon, nodes);
        var query = EmitLinear(context, file, $"{prefix}.attn_q.weight", normalized, hiddenSize, nodes, weights);
        var key = EmitLinear(context, file, $"{prefix}.attn_k.weight", normalized, keyValueWidth, nodes, weights);
        var value = EmitLinear(context, file, $"{prefix}.attn_v.weight", normalized, keyValueWidth, nodes, weights);
        query = EmitBias(context, file, $"{prefix}.attn_q.bias", query, nodes, weights);
        key = EmitBias(context, file, $"{prefix}.attn_k.bias", key, nodes, weights);
        value = EmitBias(context, file, $"{prefix}.attn_v.bias", value, nodes, weights);
        var rope = new RopeAttributes(ropeTheta, headDimension, RotaryLayout.NeoX);
        var rotatedQuery = context.Emit(
            GraphOperationKind.Rope,
            [query.Id, position.Id],
            query.Shape,
            Float,
            nodes,
            attributes: rope);
        var rotatedKey = context.Emit(
            GraphOperationKind.Rope,
            [key.Id, position.Id],
            key.Shape,
            Float,
            nodes,
            attributes: rope);
        keyState = context.AddState(ContextMatrix(maximumContextSize, keyValueWidth));
        valueState = context.AddState(ContextMatrix(maximumContextSize, keyValueWidth));
        var stateEffect = context.AddEffect();
        context.EmitStateAppend(
            [rotatedKey.Id, value.Id, position.Id],
            [keyState, valueState],
            stateEffect,
            nodes);
        var attentionAttributes = new CausalAttentionAttributes(
            attentionHeads,
            keyValueHeads,
            headDimension,
            1.0f / MathF.Sqrt(headDimension),
            AttentionMaskKind.Causal);
        var attention = context.Emit(
            GraphOperationKind.CausalAttention,
            [rotatedQuery.Id, position.Id],
            Vector(hiddenSize),
            Float,
            nodes,
            [keyState, valueState],
            [stateEffect],
            attentionAttributes);
        var projection = EmitLinear(
            context,
            file,
            $"{prefix}.attn_output.weight",
            attention,
            hiddenSize,
            nodes,
            weights);
        return context.Emit(
            GraphOperationKind.Add,
            [hidden.Id, projection.Id],
            Vector(hiddenSize),
            Float,
            nodes);
    }

    private static GraphValue AddFeedForward(
        Qwen2GraphBuildContext context,
        GgufFile file,
        string prefix,
        GraphValue residual,
        int hiddenSize,
        int feedForwardSize,
        float normalizationEpsilon,
        ICollection<NodeId> nodes,
        ICollection<TensorId> weights)
    {
        var normWeight = AddWeight(context, file, $"{prefix}.ffn_norm.weight", nodes, weights);
        var input = EmitNormalization(context, residual, normWeight, normalizationEpsilon, nodes);
        var gate = EmitLinear(context, file, $"{prefix}.ffn_gate.weight", input, feedForwardSize, nodes, weights);
        var up = EmitLinear(context, file, $"{prefix}.ffn_up.weight", input, feedForwardSize, nodes, weights);
        var activatedGate = context.Emit(GraphOperationKind.Silu, [gate.Id], gate.Shape, Float, nodes);
        var gated = context.Emit(
            GraphOperationKind.Multiply,
            [activatedGate.Id, up.Id],
            Vector(feedForwardSize),
            Float,
            nodes);
        var down = EmitLinear(context, file, $"{prefix}.ffn_down.weight", gated, hiddenSize, nodes, weights);
        return context.Emit(
            GraphOperationKind.Add,
            [residual.Id, down.Id],
            Vector(hiddenSize),
            Float,
            nodes);
    }

    private static GraphValue EmitLinear(
        Qwen2GraphBuildContext context,
        GgufFile file,
        string weightName,
        GraphValue input,
        int outputSize,
        ICollection<NodeId> nodes,
        ICollection<TensorId> weights)
    {
        var weight = context.AddWeight(file.SourceFile, file.GetRequiredTensor(weightName), nodes, weights);
        return context.Emit(
            GraphOperationKind.QuantizedLinear,
            [input.Id, weight.Value.Id],
            Vector(outputSize),
            Float,
            nodes);
    }

    private static GraphValue EmitBias(
        Qwen2GraphBuildContext context,
        GgufFile file,
        string biasName,
        GraphValue input,
        ICollection<NodeId> nodes,
        ICollection<TensorId> weights)
    {
        var bias = context.AddWeight(file.SourceFile, file.GetRequiredTensor(biasName), nodes, weights);
        return context.Emit(GraphOperationKind.Add, [input.Id, bias.Value.Id], input.Shape, Float, nodes);
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

    private static TensorShape ContextMatrix(long maximumRows, long columns) => new(
        ShapeDimension.Bounded("Context", 1, maximumRows),
        ShapeDimension.Fixed(columns));
}
