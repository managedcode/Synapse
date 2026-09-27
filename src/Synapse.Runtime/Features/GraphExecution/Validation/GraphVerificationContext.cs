using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal sealed class GraphVerificationContext(ModelGraph graph)
{
    public ModelGraph Graph { get; } = graph;

    public Dictionary<ValueId, GraphValue> Values { get; } = [];

    public Dictionary<StateSlotId, StateSlotDescriptor> StateSlots { get; } = [];

    public Dictionary<NodeId, GraphNode> Nodes { get; } = [];

    public Dictionary<NodeId, int> NodeIndices { get; } = [];

    public Dictionary<ValueId, NodeId> ValueProducers { get; } = [];

    public Dictionary<ValueId, HashSet<NodeId>> ValueConsumers { get; } = [];

    public Dictionary<TensorId, NodeId> TensorNodes { get; } = [];

    public Dictionary<TensorId, WeightDescriptor> WeightDescriptors { get; } = [];

    public Dictionary<EffectToken, NodeId> EffectProducers { get; } = [];

    public Dictionary<NodeId, HashSet<NodeId>> Edges { get; } = [];

    public List<GraphDiagnostic> Diagnostics { get; } = [];

    public void Add(GraphDiagnosticCode code, string message, NodeId? nodeId = null) =>
        Diagnostics.Add(new GraphDiagnostic(code, message, nodeId));

    public void AddEdge(NodeId producer, NodeId consumer)
    {
        if (!Edges.TryGetValue(producer, out var consumers))
        {
            consumers = [];
            Edges.Add(producer, consumers);
        }

        _ = consumers.Add(consumer);
    }

    public void AddValueConsumer(ValueId value, NodeId consumer)
    {
        if (!ValueConsumers.TryGetValue(value, out var consumers))
        {
            consumers = [];
            ValueConsumers.Add(value, consumers);
        }

        _ = consumers.Add(consumer);
    }

    public bool IsReachable(NodeId source, NodeId target)
    {
        var pending = new Stack<NodeId>();
        var visited = new HashSet<NodeId>();
        pending.Push(source);
        while (pending.TryPop(out var current))
        {
            if (!visited.Add(current) || !Edges.TryGetValue(current, out var consumers))
            {
                continue;
            }

            if (consumers.Contains(target))
            {
                return true;
            }

            foreach (var consumer in consumers)
            {
                pending.Push(consumer);
            }
        }

        return false;
    }
}
