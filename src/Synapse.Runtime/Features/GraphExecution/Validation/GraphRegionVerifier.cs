using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphRegionVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        if (context.Graph.Regions.Count == 0)
        {
            context.Add(GraphDiagnosticCode.InvalidRegion, "The graph declares no executable regions.");
            return;
        }

        var regionIds = new HashSet<RegionId>();
        var coveredNodes = new HashSet<NodeId>();
        foreach (var region in context.Graph.Regions)
        {
            if (!regionIds.Add(region.Id))
            {
                context.Add(GraphDiagnosticCode.DuplicateId, $"Region {region.Id} is declared more than once.");
            }

            var memberNodes = ValidateMembers(context, region, coveredNodes);
            GraphRegionBoundaryVerifier.Verify(context, region, memberNodes);
            GraphRegionActivationVerifier.Verify(context, region, memberNodes);
        }

        foreach (var node in context.Nodes.Values.Where(node =>
                     !IsEntryPointPlumbing(node) && !coveredNodes.Contains(node.Id)))
        {
            context.Add(
                GraphDiagnosticCode.InvalidRegion,
                $"Executable node {node.Id} is not assigned to a region.",
                node.Id);
        }
    }

    private static HashSet<NodeId> ValidateMembers(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> coveredNodes)
    {
        var localNodes = new HashSet<NodeId>();
        if (region.Nodes.Count == 0)
        {
            context.Add(GraphDiagnosticCode.InvalidRegion, $"Region {region.Id} contains no executable nodes.");
        }

        foreach (var node in region.Nodes)
        {
            if (!context.Nodes.ContainsKey(node))
            {
                context.Add(GraphDiagnosticCode.UnknownReference, $"Region {region.Id} contains unknown node {node}.");
                continue;
            }

            if (!localNodes.Add(node))
            {
                context.Add(GraphDiagnosticCode.DuplicateId, $"Region {region.Id} repeats node {node}.", node);
                continue;
            }

            if (IsEntryPointPlumbing(context.Nodes[node]))
            {
                context.Add(
                    GraphDiagnosticCode.InvalidRegion,
                    $"Region {region.Id} contains entry-point plumbing node {node}; Input and Output nodes are not executable region members.",
                    node);
                continue;
            }

            if (!coveredNodes.Add(node))
            {
                context.Add(
                    GraphDiagnosticCode.InvalidRegion,
                    $"Node {node} belongs to more than one executable region.",
                    node);
            }
        }

        _ = localNodes.RemoveWhere(node => IsEntryPointPlumbing(context.Nodes[node]));
        return localNodes;
    }

    private static bool IsEntryPointPlumbing(GraphNode node) =>
        node.Operation is GraphOperationKind.Input or GraphOperationKind.Output;

}
