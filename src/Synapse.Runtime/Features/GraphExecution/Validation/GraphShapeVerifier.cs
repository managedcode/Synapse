using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphShapeVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        foreach (var value in context.Values.Values)
        {
            ValidateNumericType(context, value);
        }

        foreach (var node in context.Graph.Nodes)
        {
            ValidateNode(context, node);
        }
    }

    private static void ValidateNumericType(GraphVerificationContext context, GraphValue value)
    {
        var numeric = value.NumericType;
        var storageAndComputeValid = numeric.Storage switch
        {
            StorageDataType.Bool => numeric.Compute == ComputeDataType.Bool,
            StorageDataType.I32 => numeric.Compute == ComputeDataType.I32,
            StorageDataType.Fp32 or StorageDataType.Fp16 or StorageDataType.Bf16 or
                StorageDataType.BlockQ8 or StorageDataType.BlockQ4 or StorageDataType.BlockQ6 =>
                numeric.Compute is ComputeDataType.Fp32 or ComputeDataType.Fp16 or ComputeDataType.Bf16,
            _ => false,
        };
        var accumulatorValid = numeric.Compute switch
        {
            ComputeDataType.Bool or ComputeDataType.I32 => numeric.Accumulator == AccumulatorDataType.I32,
            ComputeDataType.Fp32 or ComputeDataType.Fp16 or ComputeDataType.Bf16 =>
                numeric.Accumulator is AccumulatorDataType.Fp32 or AccumulatorDataType.Fp64,
            _ => false,
        };
        if (!storageAndComputeValid || !accumulatorValid)
        {
            context.Add(
                GraphDiagnosticCode.NumericTypeMismatch,
                $"Value {value.Id} has incompatible storage {numeric.Storage} and compute {numeric.Compute} types.");
        }
    }

    private static void ValidateNode(GraphVerificationContext context, GraphNode node)
    {
        if (!Enum.IsDefined(node.Operation))
        {
            context.Add(GraphDiagnosticCode.UnsupportedOperation, $"Node {node.Id} has an unknown operation kind.", node.Id);
            return;
        }

        switch (node.Operation)
        {
            case GraphOperationKind.Input or GraphOperationKind.Constant:
                _ = RequireCounts(context, node, inputMinimum: 0, inputMaximum: 0, outputCount: 1);
                break;
            case GraphOperationKind.Output:
                _ = RequireCounts(context, node, inputMinimum: 1, inputMaximum: int.MaxValue, outputCount: 0);
                break;
            case GraphOperationKind.Add or GraphOperationKind.Multiply:
                ValidateElementwiseBinary(context, node);
                break;
            case GraphOperationKind.Linear or GraphOperationKind.QuantizedLinear:
                ValidateLinear(context, node);
                break;
            case GraphOperationKind.RmsNorm:
                ValidateNormalization(context, node, biasAllowed: false);
                break;
            case GraphOperationKind.LayerNorm:
                ValidateNormalization(context, node, biasAllowed: true);
                break;
            case GraphOperationKind.Silu or GraphOperationKind.Gelu or
                GraphOperationKind.Softmax:
                ValidateShapePreservingUnary(context, node);
                break;
            case GraphOperationKind.Rope:
                ValidateRope(context, node);
                break;
            case GraphOperationKind.CausalAttention:
                ValidateCausalAttention(context, node);
                break;
            case GraphOperationKind.StateAppend:
                ValidateStateAppend(context, node);
                break;
            case GraphOperationKind.Reshape:
                ValidateReshape(context, node);
                break;
            case GraphOperationKind.Merge:
                ValidateMerge(context, node);
                break;
            case GraphOperationKind.Embedding:
                ValidateEmbedding(context, node);
                break;
            case GraphOperationKind.Transpose or GraphOperationKind.Slice or
                GraphOperationKind.Concat or GraphOperationKind.Gather or
                GraphOperationKind.Scatter or GraphOperationKind.StateRead or
                GraphOperationKind.StateCommit or GraphOperationKind.StateRollback or
                GraphOperationKind.TopKRoute or GraphOperationKind.Branch or
                GraphOperationKind.Loop or GraphOperationKind.SelectLogits or
                GraphOperationKind.Sample:
                break;
            default:
                break;
        }
    }

    private static void ValidateRope(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 2, 2, 1) ||
            !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        if (!SameShape(inputs[0].Shape, output.Shape) || !IsPosition(inputs[1]) ||
            node.Attributes is not RopeAttributes attributes || inputs[0].Shape.Rank != 1 ||
            inputs[0].Shape.Dimensions[0].Minimum != inputs[0].Shape.Dimensions[0].Maximum ||
            inputs[0].Shape.Dimensions[0].Maximum % Math.Max(1, attributes.HeadDimension) != 0)
        {
            AddShapeMismatch(
                context,
                node,
                "RoPE expects data [heads*head_dimension], scalar I32 position, and matching output");
        }
    }

    private static void ValidateCausalAttention(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 2, 2, 1) ||
            !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        var attributes = node.Attributes as CausalAttentionAttributes;
        var queryWidth = attributes is null
            ? 0L
            : checked((long)attributes.QueryHeads * attributes.HeadDimension);
        var stateWidth = attributes is null
            ? 0L
            : checked((long)attributes.KeyValueHeads * attributes.HeadDimension);
        var stateShapeValid = node.StateReads.Count == 2 && node.StateReads.All(slotId =>
            context.StateSlots.TryGetValue(slotId, out var slot) && slot.Shape.Rank == 2 &&
            slot.Shape.Dimensions[1] == ShapeDimension.Fixed(stateWidth));
        if (attributes is null || inputs[0].Shape.Rank != 1 ||
            inputs[0].Shape.Dimensions[0] != ShapeDimension.Fixed(queryWidth) ||
            !SameShape(inputs[0].Shape, output.Shape) || !IsPosition(inputs[1]) || !stateShapeValid)
        {
            AddShapeMismatch(
                context,
                node,
                "causal attention expects query [query_heads*head_dimension], scalar I32 position, two compatible KV states, and matching output");
        }
    }

    private static void ValidateStateAppend(GraphVerificationContext context, GraphNode node)
    {
        var expectedInputs = checked(node.StateWrites.Count + 1);
        if (node.StateWrites.Count == 0 ||
            !RequireCounts(context, node, expectedInputs, expectedInputs, 0))
        {
            return;
        }

        var inputs = node.Inputs
            .Select(input => context.Values.GetValueOrDefault(input))
            .ToArray();
        if (inputs.Any(input => input is null) || !IsPosition(inputs[^1]!))
        {
            AddShapeMismatch(context, node, "state append requires one value per state slot and a final scalar I32 position");
            return;
        }

        for (var index = 0; index < node.StateWrites.Count; index++)
        {
            if (!context.StateSlots.TryGetValue(node.StateWrites[index], out var slot) ||
                !MatchesStateEntry(inputs[index]!.Shape, slot.Shape))
            {
                AddShapeMismatch(context, node, "state append value shapes must match their state entry shapes");
                return;
            }
        }
    }

    private static bool MatchesStateEntry(TensorShape value, TensorShape state) =>
        state.Rank == value.Rank + 1 && state.Dimensions.Skip(1).SequenceEqual(value.Dimensions);

    private static bool IsPosition(GraphValue value) =>
        value.NumericType.Storage == StorageDataType.I32 &&
        value.NumericType.Compute == ComputeDataType.I32 &&
        value.Shape.Rank == 1 && value.Shape.Dimensions[0] == ShapeDimension.Fixed(1);

    private static void ValidateElementwiseBinary(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 2, 2, 1) || !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        if (!SameShape(inputs[0].Shape, inputs[1].Shape) || !SameShape(inputs[0].Shape, output.Shape))
        {
            AddShapeMismatch(context, node, "element-wise inputs and output must have identical shapes");
        }
    }

    private static void ValidateLinear(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 2, 3, 1) || !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        var input = inputs[0].Shape;
        var weights = inputs[1].Shape;
        if (input.Rank != 1 || weights.Rank != 2 || output.Shape.Rank != 1 ||
            !SameDimension(input.Dimensions[0], weights.Dimensions[1]) ||
            !SameDimension(output.Shape.Dimensions[0], weights.Dimensions[0]))
        {
            AddShapeMismatch(context, node, "linear expects input [in], weights [out,in], and output [out]");
            return;
        }

        if (inputs.Count == 3 && !SameShape(inputs[2].Shape, output.Shape))
        {
            AddShapeMismatch(context, node, "linear bias must match output shape");
        }
    }

    private static void ValidateShapePreservingUnary(GraphVerificationContext context, GraphNode node)
    {
        if (RequireCounts(context, node, 1, 1, 1) && TryResolve(context, node, out var inputs, out var output) &&
            !SameShape(inputs[0].Shape, output.Shape))
        {
            AddShapeMismatch(context, node, "operation must preserve shape");
        }
    }

    private static void ValidateNormalization(
        GraphVerificationContext context,
        GraphNode node,
        bool biasAllowed)
    {
        var maximumInputs = biasAllowed ? 3 : 2;
        if (!RequireCounts(context, node, 2, maximumInputs, 1) ||
            !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        if (!SameShape(inputs[0].Shape, output.Shape) ||
            inputs.Skip(1).Any(parameter => !SameShape(parameter.Shape, output.Shape)))
        {
            AddShapeMismatch(context, node, "normalization input, parameters, and output must have identical shapes");
        }
    }

    private static void ValidateEmbedding(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 2, 2, 1) ||
            !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        var indices = inputs[0];
        var weights = inputs[1];
        if (indices.NumericType.Compute != ComputeDataType.I32 ||
            indices.Shape.Rank != 1 || indices.Shape.Dimensions[0] != ShapeDimension.Fixed(1) ||
            weights.Shape.Rank != 2 || output.Shape.Rank != 1 ||
            !SameDimension(weights.Shape.Dimensions[1], output.Shape.Dimensions[0]))
        {
            AddShapeMismatch(context, node, "embedding expects one integer index, weights [vocabulary,hidden], and output [hidden]");
        }
    }

    private static void ValidateReshape(GraphVerificationContext context, GraphNode node)
    {
        if (!RequireCounts(context, node, 1, 1, 1) || !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        if (!TryElementBounds(inputs[0].Shape, out var inputMinimum, out var inputMaximum) ||
            !TryElementBounds(output.Shape, out var outputMinimum, out var outputMaximum) ||
            inputMinimum != outputMinimum || inputMaximum != outputMaximum)
        {
            AddShapeMismatch(context, node, "reshape must preserve bounded element counts");
        }
    }

    private static void ValidateMerge(GraphVerificationContext context, GraphNode node)
    {
        if (node.MergeMode is null)
        {
            AddShapeMismatch(context, node, "merge semantics must be explicit");
            return;
        }

        if (!RequireCounts(context, node, 1, int.MaxValue, 1) ||
            !TryResolve(context, node, out var inputs, out var output))
        {
            return;
        }

        if (node.MergeMode is MergeMode.Add or MergeMode.GatedSum or MergeMode.SelectActive &&
            inputs.Any(input => !SameShape(input.Shape, output.Shape)))
        {
            AddShapeMismatch(context, node, "add/gated/select-active merge inputs and output must have identical shapes");
        }
    }

    private static bool RequireCounts(
        GraphVerificationContext context,
        GraphNode node,
        int inputMinimum,
        int inputMaximum,
        int outputCount)
    {
        if (node.Inputs.Count >= inputMinimum && node.Inputs.Count <= inputMaximum &&
            node.Outputs.Count == outputCount)
        {
            return true;
        }

        AddShapeMismatch(
            context,
            node,
            $"expected {inputMinimum}..{inputMaximum} inputs and {outputCount} outputs");
        return false;
    }

    private static bool TryResolve(
        GraphVerificationContext context,
        GraphNode node,
        out IReadOnlyList<GraphValue> inputs,
        out GraphValue output)
    {
        var resolved = new List<GraphValue>(node.Inputs.Count);
        foreach (var input in node.Inputs)
        {
            if (!context.Values.TryGetValue(input, out var value))
            {
                inputs = [];
                output = null!;
                return false;
            }

            resolved.Add(value);
        }

        inputs = resolved;
        return context.Values.TryGetValue(node.Outputs[0], out output!);
    }

    private static bool SameShape(TensorShape left, TensorShape right) =>
        left.Rank == right.Rank && left.Dimensions.Zip(right.Dimensions).All(pair =>
            SameDimension(pair.First, pair.Second));

    private static bool SameDimension(ShapeDimension left, ShapeDimension right) => left == right;

    private static bool TryElementBounds(TensorShape shape, out long minimum, out long maximum)
    {
        minimum = 1;
        maximum = 1;
        try
        {
            foreach (var dimension in shape.Dimensions)
            {
                minimum = checked(minimum * dimension.Minimum);
                maximum = checked(maximum * dimension.Maximum);
            }

            return true;
        }
        catch (OverflowException)
        {
            return false;
        }
    }

    private static void AddShapeMismatch(
        GraphVerificationContext context,
        GraphNode node,
        string detail) => context.Add(
            GraphDiagnosticCode.ShapeMismatch,
            $"Node {node.Id} ({node.Operation}) {detail}.",
            node.Id);
}
