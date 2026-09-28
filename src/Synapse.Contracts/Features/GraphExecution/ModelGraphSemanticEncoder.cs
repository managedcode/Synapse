namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

internal static class ModelGraphSemanticEncoder
{
    public static void WriteAttributes(CanonicalHashWriter writer, GraphOperationAttributes? attributes)
    {
        switch (attributes)
        {
            case null:
                writer.WriteByte(0);
                break;
            case NormalizationAttributes normalization:
                writer.WriteByte(1);
                writer.WriteSingle(normalization.Epsilon);
                break;
            case RopeAttributes rope:
                // Unscaled RoPE keeps tag 2 so existing fingerprints are unchanged; a scaled profile uses tag 5.
                writer.WriteByte(rope.Scaling is null ? (byte)2 : (byte)5);
                writer.WriteSingle(rope.Theta);
                writer.WriteInt32(rope.HeadDimension);
                writer.WriteInt32((int)rope.Layout);
                if (rope.Scaling is { } scaling)
                {
                    writer.WriteInt32((int)scaling.Kind);
                    writer.WriteSingle(scaling.Factor);
                    writer.WriteInt32(scaling.OriginalContextLength);
                }

                break;
            case CausalAttentionAttributes attention:
                writer.WriteByte(3);
                writer.WriteInt32(attention.QueryHeads);
                writer.WriteInt32(attention.KeyValueHeads);
                writer.WriteInt32(attention.HeadDimension);
                writer.WriteSingle(attention.Scale);
                writer.WriteInt32((int)attention.Mask);
                writer.WriteBoolean(attention.HandlesPositionHoles);
                break;
            case TopKRouteAttributes route:
                writer.WriteByte(4);
                writer.WriteInt32(route.K);
                writer.WriteInt32((int)route.Axis);
                writer.WriteInt32((int)route.TiePolicy);
                WriteOptionalInt32(writer, route.Capacity);
                break;
            default:
                throw new NotSupportedException(
                    $"Operation attributes '{attributes.GetType().FullName}' have no canonical encoding.");
        }
    }

    public static void WriteActivation(CanonicalHashWriter writer, RegionActivation activation)
    {
        WriteDecision(writer, activation.Decision);
        WriteProvenance(writer, activation.Provenance);
        WriteSkip(writer, activation.Skip);
    }

    private static void WriteDecision(CanonicalHashWriter writer, ActivationDecision decision)
    {
        switch (decision)
        {
            case AlwaysActive:
                writer.WriteByte(1);
                break;
            case PredicateDecision predicate:
                writer.WriteByte(2);
                writer.WriteUInt32(predicate.Predicate.Value);
                writer.WriteInt32((int)predicate.Scope);
                break;
            case RouteSlotDecision route:
                writer.WriteByte(3);
                writer.WriteUInt32(route.Route.Value);
                writer.WriteInt32(route.Slot);
                writer.WriteInt32((int)route.Scope);
                break;
            case ProfileDecision profile:
                writer.WriteByte(4);
                writer.WriteString(profile.ProfileKey);
                break;
            default:
                throw Unsupported(decision, "Activation decision");
        }
    }

    private static void WriteProvenance(CanonicalHashWriter writer, EligibilityProvenance provenance)
    {
        switch (provenance)
        {
            case StructuralProvenance:
                writer.WriteByte(1);
                break;
            case ProgrammedProvenance:
                writer.WriteByte(2);
                break;
            case TrainedPolicyProvenance trained:
                writer.WriteByte(3);
                writer.WriteString(trained.PolicyHash.Value);
                break;
            case ApproximateProvenance approximate:
                writer.WriteByte(4);
                writer.WriteString(approximate.EvaluationHash.Value);
                break;
            default:
                throw Unsupported(provenance, "Eligibility provenance");
        }
    }

    private static void WriteSkip(CanonicalHashWriter writer, SkipSemantics skip)
    {
        switch (skip)
        {
            case NotSkippable:
                writer.WriteByte(1);
                break;
            case OutputsAbsent:
                writer.WriteByte(2);
                break;
            case BypassOutputs bypass:
                writer.WriteByte(3);
                var mappings = bypass.Map
                    .OrderBy(item => item.Output.Value)
                    .ThenBy(item => item.Input.Value)
                    .ToArray();
                writer.WriteCount(mappings.Length);
                foreach (var mapping in mappings)
                {
                    writer.WriteUInt32(mapping.Output.Value);
                    writer.WriteUInt32(mapping.Input.Value);
                }

                break;
            default:
                throw Unsupported(skip, "Skip semantics");
        }
    }

    private static void WriteOptionalInt32(CanonicalHashWriter writer, int? value)
    {
        writer.WriteBoolean(value.HasValue);
        if (value.HasValue)
        {
            writer.WriteInt32(value.Value);
        }
    }

    private static NotSupportedException Unsupported(object value, string name) => new(
        $"{name} '{value.GetType().FullName}' has no canonical encoding.");
}
