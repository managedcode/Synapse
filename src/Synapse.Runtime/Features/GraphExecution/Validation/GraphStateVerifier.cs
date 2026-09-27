namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphStateVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        ValidateReferences(context);
        ValidateReads(context);
        ValidateWriters(context);
    }

    private static void ValidateReferences(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            foreach (var slot in node.StateReads.Concat(node.StateWrites))
            {
                if (!context.StateSlots.ContainsKey(slot))
                {
                    context.Add(
                        GraphDiagnosticCode.UnknownReference,
                        $"Node {node.Id} references unknown state slot {slot}.",
                        node.Id);
                }
            }
        }
    }

    private static void ValidateReads(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            foreach (var slotId in node.StateReads)
            {
                if (!context.StateSlots.TryGetValue(slotId, out var slot) || slot.HasInitialValue)
                {
                    continue;
                }

                var hasPrecedingWriter = context.Graph.Nodes.Any(writer =>
                    writer.StateWrites.Contains(slotId) && context.IsReachable(writer.Id, node.Id));
                if (!hasPrecedingWriter)
                {
                    context.Add(
                        GraphDiagnosticCode.UnwrittenStateRead,
                        $"Node {node.Id} reads state slot {slotId} before any serialized writer.",
                        node.Id);
                }
            }
        }
    }

    private static void ValidateWriters(GraphVerificationContext context)
    {
        foreach (var slot in context.StateSlots.Keys)
        {
            var writers = context.Graph.Nodes.Where(node => node.StateWrites.Contains(slot)).ToArray();
            for (var left = 0; left < writers.Length; left++)
            {
                for (var right = left + 1; right < writers.Length; right++)
                {
                    if (!context.IsReachable(writers[left].Id, writers[right].Id) &&
                        !context.IsReachable(writers[right].Id, writers[left].Id))
                    {
                        context.Add(
                            GraphDiagnosticCode.UnorderedStateWriters,
                            $"Nodes {writers[left].Id} and {writers[right].Id} write state slot {slot} without serialization.",
                            writers[right].Id);
                    }
                }
            }
        }
    }
}
