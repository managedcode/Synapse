using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphEntryPointVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        if (context.Graph.EntryPoints.Count == 0)
        {
            context.Add(GraphDiagnosticCode.InvalidEntryPoint, "The graph declares no entry points.");
            return;
        }

        var ids = new HashSet<EntryPointId>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var entryPoint in context.Graph.EntryPoints)
        {
            if (!ids.Add(entryPoint.Id) || string.IsNullOrWhiteSpace(entryPoint.Name) ||
                !names.Add(entryPoint.Name))
            {
                context.Add(
                    GraphDiagnosticCode.InvalidEntryPoint,
                    $"Entry point {entryPoint.Id} has a duplicate identity/name or an empty name.");
            }

            ValidateInputs(context, entryPoint);
            ValidateOutputs(context, entryPoint);
        }
    }

    private static void ValidateInputs(GraphVerificationContext context, GraphEntryPoint entryPoint)
    {
        foreach (var input in entryPoint.Inputs)
        {
            if (!context.Values.ContainsKey(input) ||
                !context.ValueProducers.TryGetValue(input, out var producer) ||
                !context.Nodes.TryGetValue(producer, out var node) ||
                node.Operation != GraphOperationKind.Input)
            {
                context.Add(
                    GraphDiagnosticCode.InvalidEntryPoint,
                    $"Entry point {entryPoint.Id} input {input} is not produced by an Input node.");
            }
        }
    }

    private static void ValidateOutputs(GraphVerificationContext context, GraphEntryPoint entryPoint)
    {
        foreach (var output in entryPoint.Outputs)
        {
            var hasOutputNode = context.Graph.Nodes.Any(node =>
                node.Operation == GraphOperationKind.Output && node.Inputs.Contains(output));
            if (!context.Values.ContainsKey(output) ||
                !context.ValueProducers.ContainsKey(output) || !hasOutputNode)
            {
                context.Add(
                    GraphDiagnosticCode.InvalidEntryPoint,
                    $"Entry point {entryPoint.Id} output {output} is not produced and exported by an Output node.");
            }
        }
    }
}
