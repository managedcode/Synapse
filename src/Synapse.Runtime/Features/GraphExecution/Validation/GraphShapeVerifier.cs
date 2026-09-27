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
                StorageDataType.BlockQ8 or StorageDataType.BlockQ4 =>
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
                GraphOperationKind.Softmax or GraphOperationKind.Rope:
                ValidateShapePreservingUnary(context, node);
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
                GraphOperationKind.Scatter or GraphOperationKind.CausalAttention or
                GraphOperationKind.StateRead or GraphOperationKind.StateAppend or
                GraphOperationKind.StateCommit or GraphOperationKind.StateRollback or
                GraphOperationKind.TopKRoute or GraphOperationKind.Branch or
                GraphOperationKind.Loop or GraphOperationKind.SelectLogits or
                GraphOperationKind.Sample:
                ValidateExtendedOperation(context, node);
                break;
            default:
                break;
        }
    }

    private static void ValidateExtendedOperation(GraphVerificationContext context, GraphNode node)
    {
        if (node.Operation == GraphOperationKind.CausalAttention)
        {
            ValidateShapePreservingUnary(context, node);
        }
        else if (node.Operation == GraphOperationKind.StateAppend)
        {
            _ = RequireCounts(context, node, 1, int.MaxValue, 0);
        }
    }

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

        if (node.MergeMode is MergeMode.Add or MergeMode.GatedSum &&
            inputs.Any(input => !SameShape(input.Shape, output.Shape)))
        {
            AddShapeMismatch(context, node, "add/gated merge inputs and output must have identical shapes");
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
