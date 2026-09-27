namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

internal sealed class ModelGraphCanonicalEncoder(CanonicalHashWriter writer)
{
    private const uint EncodingVersion = 1;

    public void Write(ModelGraph graph)
    {
        writer.WriteString("ManagedCode.Synapse.ModelGraph");
        writer.WriteUInt32(EncodingVersion);
        writer.WriteUInt16(graph.GraphVersion.Major);
        writer.WriteUInt16(graph.GraphVersion.Minor);
        writer.WriteUInt16(graph.OpSetVersion.Major);
        writer.WriteUInt16(graph.OpSetVersion.Minor);
        WriteValues([.. graph.Values.OrderBy(value => value.Id.Value)]);
        WriteNodes([.. graph.Nodes.OrderBy(node => node.Id.Value)]);
        WriteStateSlots([.. graph.StateSlots.OrderBy(slot => slot.Id.Value)]);
        WriteEntryPoints([.. graph.EntryPoints.OrderBy(entry => entry.Id.Value)]);
        WriteRegions([.. graph.Regions.OrderBy(region => region.Id.Value)]);
    }

    private void WriteValues(IReadOnlyList<GraphValue> values)
    {
        writer.WriteCount(values.Count);
        foreach (var value in values)
        {
            writer.WriteUInt32(value.Id.Value);
            WriteShape(value.Shape);
            WriteNumericType(value.NumericType);
        }
    }

    private void WriteNodes(IReadOnlyList<GraphNode> nodes)
    {
        writer.WriteCount(nodes.Count);
        foreach (var node in nodes)
        {
            writer.WriteUInt32(node.Id.Value);
            writer.WriteInt32((int)node.Operation);
            WriteIds(node.Inputs, id => id.Value);
            WriteIds(node.Outputs, id => id.Value);
            WriteIds(node.StateReads, id => id.Value);
            WriteIds(node.StateWrites, id => id.Value);
            WriteIds(node.EffectInputs, id => id.Value);
            WriteIds(node.EffectOutputs, id => id.Value);
            WriteOptionalEnum(node.MergeMode);
            WriteLoop(node.Loop);
            WriteOptionalUInt32(node.Tensor?.Value);
            WriteAttributes(node.Attributes);
        }
    }

    private void WriteStateSlots(IReadOnlyList<StateSlotDescriptor> stateSlots)
    {
        writer.WriteCount(stateSlots.Count);
        foreach (var slot in stateSlots)
        {
            writer.WriteUInt32(slot.Id.Value);
            WriteShape(slot.Shape);
            WriteNumericType(slot.NumericType);
            writer.WriteBoolean(slot.HasInitialValue);
        }
    }

    private void WriteEntryPoints(IReadOnlyList<GraphEntryPoint> entryPoints)
    {
        writer.WriteCount(entryPoints.Count);
        foreach (var entryPoint in entryPoints)
        {
            writer.WriteUInt32(entryPoint.Id.Value);
            writer.WriteString(entryPoint.Name);
            WriteIds(entryPoint.Inputs, id => id.Value);
            WriteIds(entryPoint.Outputs, id => id.Value);
        }
    }

    private void WriteRegions(IReadOnlyList<RegionDescriptor> regions)
    {
        writer.WriteCount(regions.Count);
        foreach (var region in regions)
        {
            writer.WriteUInt32(region.Id.Value);
            WriteSortedIds(region.Nodes, id => id.Value);
            WriteSortedIds(region.Inputs, id => id.Value);
            WriteSortedIds(region.Outputs, id => id.Value);
            WriteSortedIds(region.RequiredWeights, id => id.Value);
            WriteSortedIds(region.StateReads, id => id.Value);
            WriteSortedIds(region.StateWrites, id => id.Value);
            WriteEligibility(region.Eligibility);
            WriteStrings([.. region.SemanticAnnotations.Order(StringComparer.Ordinal)]);
        }
    }

    private void WriteShape(TensorShape shape)
    {
        writer.WriteCount(shape.Dimensions.Count);
        foreach (var dimension in shape.Dimensions)
        {
            writer.WriteNullableString(dimension.Symbol);
            writer.WriteInt64(dimension.Minimum);
            writer.WriteInt64(dimension.Maximum);
        }
    }

    private void WriteNumericType(NumericType numericType)
    {
        writer.WriteInt32((int)numericType.Storage);
        writer.WriteInt32((int)numericType.Compute);
        writer.WriteInt32((int)numericType.Accumulator);
    }

    private void WriteLoop(LoopDescriptor? loop)
    {
        writer.WriteBoolean(loop is not null);
        if (loop is null)
        {
            return;
        }

        writer.WriteInt32(loop.MaximumIterations);
        WriteIds(loop.CarriedInputs, id => id.Value);
        WriteIds(loop.CarriedOutputs, id => id.Value);
    }

    private void WriteAttributes(GraphOperationAttributes? attributes)
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
                writer.WriteByte(2);
                writer.WriteSingle(rope.Theta);
                writer.WriteInt32(rope.HeadDimension);
                writer.WriteInt32((int)rope.Layout);
                break;
            case CausalAttentionAttributes attention:
                writer.WriteByte(3);
                writer.WriteInt32(attention.QueryHeads);
                writer.WriteInt32(attention.KeyValueHeads);
                writer.WriteInt32(attention.HeadDimension);
                writer.WriteSingle(attention.Scale);
                writer.WriteInt32((int)attention.Mask);
                break;
            default:
                throw new NotSupportedException(
                    $"Operation attributes '{attributes.GetType().FullName}' have no canonical encoding.");
        }
    }

    private void WriteEligibility(ExecutionEligibility eligibility)
    {
        switch (eligibility)
        {
            case AlwaysRequiredEligibility:
                writer.WriteByte(1);
                break;
            case GraphPredicateEligibility predicate:
                writer.WriteByte(2);
                writer.WriteUInt32(predicate.Predicate.Value);
                break;
            case TrainedRouteEligibility trained:
                writer.WriteByte(3);
                writer.WriteString(trained.PolicyHash.Value);
                break;
            case ApproximateProfileEligibility approximate:
                writer.WriteByte(4);
                writer.WriteString(approximate.EvaluationHash.Value);
                break;
            default:
                throw new NotSupportedException(
                    $"Execution eligibility '{eligibility.GetType().FullName}' has no canonical encoding.");
        }
    }

    private void WriteIds<T>(IReadOnlyList<T> ids, Func<T, uint> select)
    {
        writer.WriteCount(ids.Count);
        foreach (var id in ids)
        {
            writer.WriteUInt32(select(id));
        }
    }

    private void WriteSortedIds<T>(IReadOnlyList<T> ids, Func<T, uint> select) =>
        WriteIds([.. ids.OrderBy(select)], select);

    private void WriteStrings(IReadOnlyList<string> values)
    {
        writer.WriteCount(values.Count);
        foreach (var value in values)
        {
            writer.WriteString(value);
        }
    }

    private void WriteOptionalEnum<T>(T? value)
        where T : struct, Enum
    {
        writer.WriteBoolean(value.HasValue);
        if (value.HasValue)
        {
            writer.WriteInt32(Convert.ToInt32(
                value.Value,
                System.Globalization.CultureInfo.InvariantCulture));
        }
    }

    private void WriteOptionalUInt32(uint? value)
    {
        writer.WriteBoolean(value.HasValue);
        if (value.HasValue)
        {
            writer.WriteUInt32(value.Value);
        }
    }
}
