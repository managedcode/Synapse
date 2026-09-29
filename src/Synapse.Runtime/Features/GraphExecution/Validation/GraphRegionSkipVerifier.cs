using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphRegionSkipVerifier
{
    public static void Verify(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes)
    {
        switch (region.Activation.Skip)
        {
            case NotSkippable:
                break;
            case OutputsAbsent:
                ValidateAbsentOutputs(context, region, memberNodes);
                break;
            case BypassOutputs bypass:
                ValidateBypass(context, region, bypass);
                break;
            default:
                AddInvalid(context, region, "skip semantics are unknown");
                break;
        }

        ValidateState(context, region);
    }

    private static void ValidateAbsentOutputs(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes)
    {
        foreach (var output in region.Outputs)
        {
            var consumers = context.ValueConsumers.GetValueOrDefault(output) ?? [];
            var intolerant = consumers
                .Where(consumer => !memberNodes.Contains(consumer))
                .Select(consumer => context.Nodes[consumer])
                .Any(node => node.Operation != GraphOperationKind.Merge ||
                    node.MergeMode is not (MergeMode.Add or MergeMode.SelectActive));
            if (intolerant || context.Graph.EntryPoints.Any(entry => entry.Outputs.Contains(output)))
            {
                AddInvalid(context, region, $"absent output {output} has a consumer that cannot tolerate absence");
            }
        }
    }

    private static void ValidateBypass(
        GraphVerificationContext context,
        RegionDescriptor region,
        BypassOutputs bypass)
    {
        var outputs = bypass.Map.Select(item => item.Output).ToArray();
        if (outputs.Length != region.Outputs.Count || outputs.Distinct().Count() != outputs.Length ||
            !outputs.ToHashSet().SetEquals(region.Outputs))
        {
            AddInvalid(context, region, "bypass map must contain each region output exactly once");
            return;
        }

        foreach (var mapping in bypass.Map)
        {
            if (!region.Inputs.Contains(mapping.Input) ||
                !context.Values.TryGetValue(mapping.Output, out var output) ||
                !context.Values.TryGetValue(mapping.Input, out var input) ||
                output.NumericType != input.NumericType || !SameShape(output.Shape, input.Shape))
            {
                AddInvalid(context, region, $"bypass {mapping.Output}->{mapping.Input} must use a shape-compatible region input");
            }
        }
    }

    private static void ValidateState(GraphVerificationContext context, RegionDescriptor region)
    {
        if (region.Activation.Skip is NotSkippable)
        {
            return;
        }

        foreach (var slotId in region.StateWrites)
        {
            if (SessionWideInternalState(context, region, slotId))
            {
                continue;
            }

            if (!context.StateSlots.TryGetValue(slotId, out var slot) || !slot.PositionHolesAllowed)
            {
                AddInvalid(context, region, $"skippable state writer requires position holes for slot {slotId}");
                continue;
            }

            var readersAreHoleAware = context.Graph.Nodes
                .Where(node => node.StateReads.Contains(slotId))
                .All(node => node.Attributes is CausalAttentionAttributes { HandlesPositionHoles: true });
            if (!readersAreHoleAware)
            {
                AddInvalid(context, region, $"every reader of position-hole slot {slotId} must be hole-aware");
            }
        }
    }

    /// <summary>
    /// A profile decision holds for the whole session (ADR-019), so its region runs for every position or for none.
    /// State that only the region itself reads can then never show a position hole.
    /// </summary>
    private static bool SessionWideInternalState(GraphVerificationContext context, RegionDescriptor region, StateSlotId slotId) =>
        region.Activation.Decision is ProfileDecision &&
        context.Graph.Nodes.Where(node => node.StateReads.Contains(slotId)).All(node => region.Nodes.Contains(node.Id));

    private static bool SameShape(TensorShape left, TensorShape right) =>
        left.Rank == right.Rank && left.Dimensions.SequenceEqual(right.Dimensions);

    private static void AddInvalid(GraphVerificationContext context, RegionDescriptor region, string detail) =>
        context.Add(GraphDiagnosticCode.InvalidRegion, $"Region {region.Id} {detail}.");
}
