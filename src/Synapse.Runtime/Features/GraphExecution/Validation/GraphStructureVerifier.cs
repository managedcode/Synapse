using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphStructureVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        IndexProducers(context);
        ConnectConsumers(context);
        ValidateLoops(context);
        DetectCycles(context);
    }

    private static void IndexProducers(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            foreach (var output in node.Outputs)
            {
                if (!context.Values.ContainsKey(output))
                {
                    context.Add(GraphDiagnosticCode.UnknownReference, $"Node {node.Id} produces unknown value {output}.", node.Id);
                }
                else if (!context.ValueProducers.TryAdd(output, node.Id))
                {
                    context.Add(GraphDiagnosticCode.MultipleProducers, $"Value {output} has multiple producers.", node.Id);
                }
            }

            foreach (var effect in node.EffectOutputs)
            {
                if (!context.EffectProducers.TryAdd(effect, node.Id))
                {
                    context.Add(GraphDiagnosticCode.MultipleProducers, $"Effect {effect} has multiple producers.", node.Id);
                }
            }
        }
    }

    private static void ConnectConsumers(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            foreach (var input in node.Inputs)
            {
                ConnectValue(context, node, input);
            }

            foreach (var effect in node.EffectInputs)
            {
                ConnectEffect(context, node, effect);
            }
        }
    }

    private static void ConnectValue(GraphVerificationContext context, GraphNode consumer, ValueId input)
    {
        if (!context.Values.ContainsKey(input))
        {
            context.Add(GraphDiagnosticCode.UnknownReference, $"Node {consumer.Id} consumes unknown value {input}.", consumer.Id);
            return;
        }

        if (!context.ValueProducers.TryGetValue(input, out var producer))
        {
            context.Add(GraphDiagnosticCode.MissingProducer, $"Value {input} has no producer.", consumer.Id);
            return;
        }

        context.AddValueConsumer(input, consumer.Id);
        ConnectOrdered(context, producer, consumer.Id, $"value {input}");
    }

    private static void ConnectEffect(GraphVerificationContext context, GraphNode consumer, EffectToken effect)
    {
        if (!context.EffectProducers.TryGetValue(effect, out var producer))
        {
            context.Add(GraphDiagnosticCode.UnknownReference, $"Effect {effect} has no producer.", consumer.Id);
            return;
        }

        ConnectOrdered(context, producer, consumer.Id, $"effect {effect}");
    }

    private static void ConnectOrdered(
        GraphVerificationContext context,
        NodeId producer,
        NodeId consumer,
        string dependency)
    {
        context.AddEdge(producer, consumer);
        if (context.NodeIndices.TryGetValue(producer, out var producerIndex) &&
            context.NodeIndices.TryGetValue(consumer, out var consumerIndex) &&
            producerIndex >= consumerIndex)
        {
            context.Add(
                GraphDiagnosticCode.ProducerAfterConsumer,
                $"Producer {producer} for {dependency} does not precede consumer {consumer}.",
                consumer);
        }
    }

    private static void ValidateLoops(GraphVerificationContext context)
    {
        foreach (var node in context.Graph.Nodes)
        {
            if (node.Operation == GraphOperationKind.Loop)
            {
                if (node.Loop is null || node.Loop.MaximumIterations <= 0 ||
                    node.Loop.CarriedInputs.Count != node.Loop.CarriedOutputs.Count)
                {
                    context.Add(GraphDiagnosticCode.InvalidLoop, $"Loop node {node.Id} has an invalid bound or carried-state schema.", node.Id);
                }
            }
            else if (node.Loop is not null)
            {
                context.Add(GraphDiagnosticCode.InvalidLoop, $"Non-loop node {node.Id} declares a loop contract.", node.Id);
            }
        }
    }

    private static void DetectCycles(GraphVerificationContext context)
    {
        var states = new Dictionary<NodeId, VisitState>();
        foreach (var node in context.Nodes.Keys)
        {
            if (Visit(context, node, states))
            {
                context.Add(GraphDiagnosticCode.IllegalCycle, "The graph contains an unstructured dependency cycle.", node);
                return;
            }
        }
    }

    private static bool Visit(
        GraphVerificationContext context,
        NodeId node,
        Dictionary<NodeId, VisitState> states)
    {
        if (states.GetValueOrDefault(node) == VisitState.Visiting)
        {
            return true;
        }

        if (states.GetValueOrDefault(node) == VisitState.Visited)
        {
            return false;
        }

        states[node] = VisitState.Visiting;
        if (context.Edges.TryGetValue(node, out var consumers))
        {
            foreach (var consumer in consumers)
            {
                if (Visit(context, consumer, states))
                {
                    return true;
                }
            }
        }

        states[node] = VisitState.Visited;
        return false;
    }

    private enum VisitState
    {
        Unvisited,
        Visiting,
        Visited,
    }
}
