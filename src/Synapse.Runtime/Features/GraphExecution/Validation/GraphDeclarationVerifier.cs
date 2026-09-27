using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphDeclarationVerifier
{
    public static void Index(GraphVerificationContext context)
    {
        IndexValues(context);
        IndexStateSlots(context);
        IndexNodes(context);
    }

    private static void IndexValues(GraphVerificationContext context)
    {
        foreach (var value in context.Graph.Values)
        {
            if (!context.Values.TryAdd(value.Id, value))
            {
                context.Add(GraphDiagnosticCode.DuplicateId, $"Value {value.Id} is declared more than once.");
            }

            ValidateShape(context, value.Shape, $"Value {value.Id}");
        }
    }

    private static void IndexStateSlots(GraphVerificationContext context)
    {
        foreach (var slot in context.Graph.StateSlots)
        {
            if (!context.StateSlots.TryAdd(slot.Id, slot))
            {
                context.Add(GraphDiagnosticCode.DuplicateId, $"State slot {slot.Id} is declared more than once.");
            }

            ValidateShape(context, slot.Shape, $"State slot {slot.Id}");
        }
    }

    private static void IndexNodes(GraphVerificationContext context)
    {
        for (var index = 0; index < context.Graph.Nodes.Count; index++)
        {
            var node = context.Graph.Nodes[index];
            if (!context.Nodes.TryAdd(node.Id, node))
            {
                context.Add(GraphDiagnosticCode.DuplicateId, $"Node {node.Id} is declared more than once.", node.Id);
                continue;
            }

            context.NodeIndices.Add(node.Id, index);
            context.Edges.Add(node.Id, []);
        }
    }

    private static void ValidateShape(
        GraphVerificationContext context,
        TensorShape shape,
        string owner)
    {
        if (shape.Rank is 0 or > 8)
        {
            context.Add(GraphDiagnosticCode.ShapeMismatch, $"{owner} rank {shape.Rank} is outside [1, 8].");
            return;
        }

        var maximumElements = 1L;
        foreach (var dimension in shape.Dimensions)
        {
            if (!IsValidDimension(dimension))
            {
                context.Add(GraphDiagnosticCode.ShapeMismatch, $"{owner} has an invalid bounded dimension.");
                return;
            }

            try
            {
                maximumElements = checked(maximumElements * dimension.Maximum);
            }
            catch (OverflowException)
            {
                context.Add(GraphDiagnosticCode.ShapeMismatch, $"{owner} maximum element count overflows Int64.");
                return;
            }
        }
    }

    private static bool IsValidDimension(ShapeDimension dimension) =>
        dimension.Minimum > 0 &&
        dimension.Maximum >= dimension.Minimum &&
        (dimension.IsSymbolic
            ? !string.IsNullOrWhiteSpace(dimension.Symbol)
            : dimension.Minimum == dimension.Maximum);
}
