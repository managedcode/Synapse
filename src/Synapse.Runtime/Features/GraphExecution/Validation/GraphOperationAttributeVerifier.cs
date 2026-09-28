using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphOperationAttributeVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            if (node.Operation is not (GraphOperationKind.RmsNorm or GraphOperationKind.LayerNorm or
                GraphOperationKind.Rope or GraphOperationKind.CausalAttention or GraphOperationKind.TopKRoute) &&
                node.Attributes is not null)
            {
                AddInvalid(context, node, "does not accept operation attributes");
            }

            switch (node.Operation)
            {
                case GraphOperationKind.RmsNorm or GraphOperationKind.LayerNorm:
                    ValidateNormalization(context, node);
                    break;
                case GraphOperationKind.Rope:
                    ValidateRope(context, node);
                    break;
                case GraphOperationKind.CausalAttention:
                    ValidateAttention(context, node);
                    break;
                case GraphOperationKind.TopKRoute:
                    ValidateTopKRoute(context, node);
                    break;
                case GraphOperationKind.Input:
                    break;
                case GraphOperationKind.Constant:
                    break;
                case GraphOperationKind.Embedding:
                    break;
                case GraphOperationKind.Linear:
                    break;
                case GraphOperationKind.QuantizedLinear:
                    break;
                case GraphOperationKind.Add:
                    break;
                case GraphOperationKind.Multiply:
                    break;
                case GraphOperationKind.Silu:
                    break;
                case GraphOperationKind.Gelu:
                    break;
                case GraphOperationKind.Reshape:
                    break;
                case GraphOperationKind.Transpose:
                    break;
                case GraphOperationKind.Slice:
                    break;
                case GraphOperationKind.Concat:
                    break;
                case GraphOperationKind.Gather:
                    break;
                case GraphOperationKind.Scatter:
                    break;
                case GraphOperationKind.Softmax:
                    break;
                case GraphOperationKind.StateRead:
                    break;
                case GraphOperationKind.StateAppend:
                    break;
                case GraphOperationKind.StateCommit:
                    break;
                case GraphOperationKind.StateRollback:
                    break;
                case GraphOperationKind.Branch:
                    break;
                case GraphOperationKind.Merge:
                    break;
                case GraphOperationKind.Loop:
                    break;
                case GraphOperationKind.SelectLogits:
                    break;
                case GraphOperationKind.Sample:
                    break;
                case GraphOperationKind.Output:
                    break;
                default:
                    break;
            }
        }
    }

    private static void ValidateNormalization(GraphVerificationContext context, GraphNode node)
    {
        if (node.Attributes is not NormalizationAttributes attributes ||
            !float.IsFinite(attributes.Epsilon) || attributes.Epsilon <= 0)
        {
            AddInvalid(context, node, "requires a positive finite normalization epsilon");
        }
    }

    private static void ValidateRope(GraphVerificationContext context, GraphNode node)
    {
        if (node.Attributes is not RopeAttributes attributes ||
            !float.IsFinite(attributes.Theta) || attributes.Theta <= 0 ||
            attributes.HeadDimension <= 0 || (attributes.HeadDimension & 1) != 0 ||
            !Enum.IsDefined(attributes.Layout) ||
            (attributes.Scaling is { } scaling && (!Enum.IsDefined(scaling.Kind) || !float.IsFinite(scaling.Factor) ||
                scaling.Factor <= 1 || scaling.OriginalContextLength <= 0)))
        {
            AddInvalid(context, node, "requires valid theta, even head dimension, rotary layout, and scaling profile");
        }
    }

    private static void ValidateAttention(GraphVerificationContext context, GraphNode node)
    {
        if (node.Attributes is not CausalAttentionAttributes attributes ||
            attributes.QueryHeads <= 0 || attributes.KeyValueHeads <= 0 ||
            attributes.QueryHeads % attributes.KeyValueHeads != 0 ||
            attributes.HeadDimension <= 0 || !float.IsFinite(attributes.Scale) || attributes.Scale <= 0 ||
            !Enum.IsDefined(attributes.Mask))
        {
            AddInvalid(
                context,
                node,
                "requires compatible query/KV heads, head dimension, positive finite scale, and mask");
        }
    }

    private static void ValidateTopKRoute(GraphVerificationContext context, GraphNode node)
    {
        if (node.Attributes is not TopKRouteAttributes attributes ||
            attributes.K <= 0 || !Enum.IsDefined(attributes.Axis) ||
            !Enum.IsDefined(attributes.TiePolicy) || attributes.Capacity is <= 0)
        {
            AddInvalid(context, node, "requires positive k, valid axis and tie policy, and optional positive capacity");
        }
    }

    private static void AddInvalid(
        GraphVerificationContext context,
        GraphNode node,
        string detail) => context.Add(
        GraphDiagnosticCode.InvalidOperationAttributes,
        $"Node {node.Id} ({node.Operation}) {detail}.",
        node.Id);
}
