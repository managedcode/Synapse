using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

internal static class GraphWeightVerifier
{
    public static void Verify(GraphVerificationContext context)
    {
        foreach (var (tensorId, nodeId) in context.TensorNodes)
        {
            if (!context.WeightDescriptors.ContainsKey(tensorId))
            {
                AddInvalid(context, tensorId, $"Constant node {nodeId} has no weight descriptor", nodeId);
            }
        }

        foreach (var descriptor in context.WeightDescriptors.Values)
        {
            if (!context.TensorNodes.TryGetValue(descriptor.Id, out var nodeId))
            {
                AddInvalid(context, descriptor.Id, "does not resolve to a Constant node");
                continue;
            }

            ValidateSource(context, descriptor, nodeId);
            ValidateValue(context, descriptor, context.Nodes[nodeId], nodeId);
        }
    }

    private static void ValidateSource(
        GraphVerificationContext context,
        WeightDescriptor descriptor,
        NodeId nodeId)
    {
        var range = descriptor.Source;
        if (!IsSafeRelativeFile(range.File) || range.Offset < 0 || range.Length <= 0 ||
            !RangeEndFits(range.Offset, range.Length) || !Enum.IsDefined(descriptor.Encoding))
        {
            AddInvalid(context, descriptor.Id, "has an invalid package file, byte range, or encoding", nodeId);
        }

        if (descriptor.ContentHash is { } contentHash && !IsSha256(contentHash.Value))
        {
            AddInvalid(context, descriptor.Id, "has an invalid content hash", nodeId);
        }
    }

    private static void ValidateValue(
        GraphVerificationContext context,
        WeightDescriptor descriptor,
        GraphNode node,
        NodeId nodeId)
    {
        if (node.Outputs.Count != 1 || !context.Values.TryGetValue(node.Outputs[0], out var value))
        {
            return;
        }

        if (!SameShape(descriptor.LogicalShape, value.Shape) ||
            !EncodingMatchesStorage(descriptor.Encoding, value.NumericType.Storage))
        {
            AddInvalid(
                context,
                descriptor.Id,
                "logical shape or encoding does not match its Constant value",
                nodeId);
        }
    }

    private static bool IsSafeRelativeFile(string file)
    {
        if (string.IsNullOrWhiteSpace(file))
        {
            return false;
        }

        var normalized = file.Replace('\\', '/');
        var hasWindowsDrivePrefix = normalized.Length >= 2 &&
            char.IsAsciiLetter(normalized[0]) && normalized[1] == ':';
        if (Path.IsPathFullyQualified(file) || normalized.StartsWith('/') || hasWindowsDrivePrefix)
        {
            return false;
        }

        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        return segments.Length > 0 && segments.All(segment => segment is not ("." or ".."));
    }

    private static bool RangeEndFits(long offset, long length)
    {
        try
        {
            _ = checked(offset + length);
            return true;
        }
        catch (OverflowException)
        {
            return false;
        }
    }

    private static bool IsSha256(string value) => value is { Length: 64 } && value.All(character =>
        character is (>= '0' and <= '9') or (>= 'a' and <= 'f'));

    private static bool SameShape(TensorShape left, TensorShape right) =>
        left.Dimensions.SequenceEqual(right.Dimensions);

    private static bool EncodingMatchesStorage(WeightEncoding encoding, StorageDataType storage) => encoding switch
    {
        WeightEncoding.Fp32 => storage == StorageDataType.Fp32,
        WeightEncoding.GgmlQ8Zero => storage == StorageDataType.BlockQ8,
        _ => false,
    };

    private static void AddInvalid(
        GraphVerificationContext context,
        TensorId tensorId,
        string detail,
        NodeId? nodeId = null) => context.Add(
        GraphDiagnosticCode.InvalidWeightDescriptor,
        $"Weight descriptor {tensorId} {detail}.",
        nodeId);
}
