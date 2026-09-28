using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphRegionActivationVerifier
{
    public static void Verify(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes)
    {
        ValidateDecision(context, region, memberNodes);
        ValidateProvenance(context, region);
        GraphRegionSkipVerifier.Verify(context, region, memberNodes);
    }

    private static void ValidateDecision(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes)
    {
        switch (region.Activation.Decision)
        {
            case AlwaysActive:
                if (region.Activation.Skip is not NotSkippable)
                {
                    AddInvalid(context, region, "always-active regions must be not-skippable");
                }

                break;
            case PredicateDecision predicate:
                ValidateScope(context, region, predicate.Scope);
                ValidateDecisionProducer(context, region, memberNodes, predicate.Predicate);
                ValidatePredicate(context, region, predicate);
                ValidateCausality(context, region, predicate.Predicate, predicate.Scope);
                break;
            case RouteSlotDecision route:
                ValidateScope(context, region, route.Scope);
                ValidateDecisionProducer(context, region, memberNodes, route.Route);
                ValidateRoute(context, region, route);
                ValidateCausality(context, region, route.Route, route.Scope);
                break;
            case ProfileDecision profile:
                if (string.IsNullOrWhiteSpace(profile.ProfileKey))
                {
                    AddInvalid(context, region, "profile decision key must not be empty");
                }

                break;
            default:
                AddInvalid(context, region, "activation decision is unknown");
                break;
        }

        if (region.Activation.Decision is not AlwaysActive && region.Activation.Skip is NotSkippable)
        {
            AddInvalid(context, region, "conditional regions must declare skippable output semantics");
        }
    }

    private static void ValidateProvenance(GraphVerificationContext context, RegionDescriptor region)
    {
        switch (region.Activation.Provenance)
        {
            case StructuralProvenance or ProgrammedProvenance:
                break;
            case TrainedPolicyProvenance trained:
                ValidateHash(context, region, trained.PolicyHash, "trained policy");
                if (region.Activation.Decision is not (PredicateDecision or RouteSlotDecision))
                {
                    AddInvalid(context, region, "trained policy provenance requires a graph decision value");
                }

                break;
            case ApproximateProvenance approximate:
                ValidateHash(context, region, approximate.EvaluationHash, "evaluation evidence");
                break;
            default:
                AddInvalid(context, region, "eligibility provenance is unknown");
                break;
        }
    }

    private static void ValidateDecisionProducer(
        GraphVerificationContext context,
        RegionDescriptor region,
        HashSet<NodeId> memberNodes,
        ValueId decision)
    {
        if (!context.ValueProducers.TryGetValue(decision, out var producer))
        {
            AddInvalid(context, region, $"decision value {decision} has no producer");
            return;
        }

        if (memberNodes.Contains(producer) || memberNodes.Any(node => context.IsReachable(node, producer)))
        {
            AddInvalid(context, region, $"decision value {decision} must be produced outside the region");
            return;
        }

        if (context.NodeIndices.TryGetValue(producer, out var producerIndex) &&
            memberNodes.Any(node => context.NodeIndices.TryGetValue(node, out var index) &&
                producerIndex >= index))
        {
            AddInvalid(context, region, $"decision value {decision} must precede every region node");
        }
    }

    private static void ValidatePredicate(
        GraphVerificationContext context,
        RegionDescriptor region,
        PredicateDecision predicate)
    {
        if (!context.Values.TryGetValue(predicate.Predicate, out var value) ||
            value.NumericType.Storage != StorageDataType.Bool || value.Shape.Rank != 1 ||
            ((predicate.Scope is RouteScope.Session or RouteScope.Step) &&
            value.Shape.Dimensions[0] != ShapeDimension.Fixed(1)))
        {
            context.Add(
                GraphDiagnosticCode.NumericTypeMismatch,
                $"Region {region.Id} predicate {predicate.Predicate} has an invalid boolean shape.");
        }
    }

    private static void ValidateRoute(
        GraphVerificationContext context,
        RegionDescriptor region,
        RouteSlotDecision route)
    {
        if (!context.ValueProducers.TryGetValue(route.Route, out var producer) ||
            !context.Nodes.TryGetValue(producer, out var node) ||
            node.Operation != GraphOperationKind.TopKRoute ||
            node.Attributes is not TopKRouteAttributes attributes ||
            route.Slot < 0 || route.Slot >= attributes.K)
        {
            AddInvalid(context, region, $"route decision {route.Route} must select a declared TopKRoute slot");
            return;
        }

        if (!context.Values.TryGetValue(route.Route, out var value) ||
            value.NumericType.Storage != StorageDataType.I32 ||
            value.NumericType.Compute != ComputeDataType.I32 ||
            value.Shape.Rank == 0 ||
            value.Shape.Dimensions[^1] != ShapeDimension.Fixed(attributes.K))
        {
            AddInvalid(context, region, $"route decision {route.Route} must be an I32 top-k index value");
        }
    }

    private static void ValidateCausality(
        GraphVerificationContext context,
        RegionDescriptor region,
        ValueId decision,
        RouteScope scope)
    {
        if (scope == RouteScope.Session || !context.ValueProducers.TryGetValue(decision, out var producer))
        {
            return;
        }

        var sequenceRoute = context.Graph.Nodes.Any(node =>
            node.Operation == GraphOperationKind.TopKRoute &&
            node.Attributes is TopKRouteAttributes { Axis: TopKRouteAxis.Sequence } &&
            (node.Id == producer || context.IsReachable(node.Id, producer)));
        if (sequenceRoute)
        {
            AddInvalid(context, region, "step/token decode routing must be causal; sequence-axis routing is non-causal");
        }
    }

    private static void ValidateScope(GraphVerificationContext context, RegionDescriptor region, RouteScope scope)
    {
        if (!Enum.IsDefined(scope))
        {
            AddInvalid(context, region, "route scope is unknown");
        }
    }

    private static void ValidateHash(
        GraphVerificationContext context,
        RegionDescriptor region,
        ContentHash hash,
        string purpose)
    {
        if (hash.Value is null || hash.Value.Length != 64 || hash.Value.Any(character =>
                character is not (>= '0' and <= '9') and not (>= 'a' and <= 'f')))
        {
            AddInvalid(context, region, $"{purpose} hash must be a lower-case 64-character SHA-256 digest");
        }
    }

    private static void AddInvalid(GraphVerificationContext context, RegionDescriptor region, string detail) =>
        context.Add(GraphDiagnosticCode.InvalidRegion, $"Region {region.Id} {detail}.");
}
