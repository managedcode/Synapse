using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphRegionBoundaryVerifier
{
    public static void Verify(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes)
    {
        ValidateDeclaredReferences(context, region);
        if (region.SemanticAnnotations.Any(string.IsNullOrWhiteSpace))
        {
            context.Add(GraphDiagnosticCode.InvalidRegion, $"Region {region.Id} contains an empty semantic annotation.");
        }

        if (memberNodes.Count == 0)
        {
            return;
        }

        var expected = Derive(context, memberNodes);
        var mismatches = new List<string>();
        Compare(region.Inputs, expected.Inputs, "inputs", mismatches);
        Compare(region.Outputs, expected.Outputs, "outputs", mismatches);
        Compare(region.RequiredWeights, expected.RequiredWeights, "required weights", mismatches);
        Compare(region.StateReads, expected.StateReads, "state reads", mismatches);
        Compare(region.StateWrites, expected.StateWrites, "state writes", mismatches);
        if (mismatches.Count > 0)
        {
            context.Add(
                GraphDiagnosticCode.RegionBoundaryMismatch,
                $"Region {region.Id} boundary does not match its member nodes: {string.Join("; ", mismatches)}.");
        }

    }

    private static DerivedRegionBoundary Derive(
        GraphVerificationContext context,
        HashSet<NodeId> memberNodes)
    {
        var inputs = new HashSet<ValueId>();
        var outputs = new HashSet<ValueId>();
        var weights = new HashSet<TensorId>();
        var stateReads = new HashSet<StateSlotId>();
        var stateWrites = new HashSet<StateSlotId>();
        var entryOutputs = context.Graph.EntryPoints
            .SelectMany(entryPoint => entryPoint.Outputs)
            .ToHashSet();
        var decisionValues = context.Graph.Regions
            .Select(region => region.Activation.Decision)
            .SelectMany(GetDecisionValues)
            .ToHashSet();

        foreach (var nodeId in memberNodes)
        {
            var node = context.Nodes[nodeId];
            foreach (var input in node.Inputs)
            {
                if (!context.ValueProducers.TryGetValue(input, out var producer) ||
                    !memberNodes.Contains(producer))
                {
                    _ = inputs.Add(input);
                }
            }

            foreach (var output in node.Outputs)
            {
                var consumedOutside = context.ValueConsumers.TryGetValue(output, out var consumers) &&
                    consumers.Any(consumer => !memberNodes.Contains(consumer));
                if (consumedOutside || entryOutputs.Contains(output) || decisionValues.Contains(output))
                {
                    _ = outputs.Add(output);
                }
            }

            if (node.Operation == GraphOperationKind.Constant && node.Tensor is { } tensor)
            {
                _ = weights.Add(tensor);
            }

            stateReads.UnionWith(node.StateReads);
            stateWrites.UnionWith(node.StateWrites);
        }

        return new DerivedRegionBoundary(inputs, outputs, weights, stateReads, stateWrites);
    }

    private static IEnumerable<ValueId> GetDecisionValues(ActivationDecision decision) => decision switch
    {
        PredicateDecision predicate => [predicate.Predicate],
        RouteSlotDecision route => [route.Route],
        _ => [],
    };

    private static void ValidateDeclaredReferences(
        GraphVerificationContext context,
        RegionDescriptor region)
    {
        foreach (var value in region.Inputs.Concat(region.Outputs))
        {
            if (!context.Values.ContainsKey(value))
            {
                context.Add(GraphDiagnosticCode.UnknownReference, $"Region {region.Id} references unknown value {value}.");
            }
        }

        foreach (var tensor in region.RequiredWeights)
        {
            if (!context.TensorNodes.ContainsKey(tensor))
            {
                context.Add(GraphDiagnosticCode.UnknownReference, $"Region {region.Id} references unknown tensor {tensor}.");
            }
        }

        foreach (var slot in region.StateReads.Concat(region.StateWrites))
        {
            if (!context.StateSlots.ContainsKey(slot))
            {
                context.Add(GraphDiagnosticCode.UnknownReference, $"Region {region.Id} references unknown state slot {slot}.");
            }
        }
    }

    private static void Compare<T>(
        IReadOnlyList<T> declared,
        HashSet<T> expected,
        string name,
        List<string> mismatches)
        where T : notnull
    {
        if (declared.Count != expected.Count || !expected.SetEquals(declared))
        {
            mismatches.Add(name);
        }
    }

    private sealed record DerivedRegionBoundary(
        HashSet<ValueId> Inputs,
        HashSet<ValueId> Outputs,
        HashSet<TensorId> RequiredWeights,
        HashSet<StateSlotId> StateReads,
        HashSet<StateSlotId> StateWrites);
}
