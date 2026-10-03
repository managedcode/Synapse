using System.Text.Json;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal static class SafeTensorGraphReader
{
    private const int MaximumGraphBytes = 4 * 1024 * 1024;

    public static ConversionGraph Read(string path, CancellationToken cancellationToken)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length is 0 or > MaximumGraphBytes)
        {
            throw new InvalidDataException($"SafeTensors graph sidecar must be inside [1, {MaximumGraphBytes}] bytes.");
        }

        var bytes = new byte[checked((int)stream.Length)];
        stream.ReadExactly(bytes);
        cancellationToken.ThrowIfCancellationRequested();
        if (bytes.AsSpan().StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
        {
            bytes = bytes[3..];
        }
        try
        {
            using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 16 });
            RejectDuplicateProperties(document.RootElement);
            var graph = JsonSerializer.Deserialize(bytes, ConversionJsonContext.Default.ConversionGraph) ??
                throw new InvalidDataException("SafeTensors graph sidecar cannot be null.");
            ValidateStructure(graph);
            return graph;
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("SafeTensors graph sidecar is not valid inert conversion graph JSON.", exception);
        }
    }

    private static void RejectDuplicateProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in value.EnumerateObject())
            {
                if (!names.Add(property.Name))
                {
                    throw new InvalidDataException($"SafeTensors graph sidecar repeats property '{property.Name}'.");
                }

                RejectDuplicateProperties(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
        {
            foreach (var element in value.EnumerateArray())
            {
                RejectDuplicateProperties(element);
            }
        }
    }

    private static void ValidateStructure(ConversionGraph graph)
    {
        if (graph.SchemaVersion != 1 || graph.Inputs is null || graph.Inputs.Length is 0 or > 512 ||
            graph.Outputs is null || graph.Outputs.Length is 0 or > 512 ||
            graph.Nodes is null || graph.Nodes.Length is 0 or > 4096)
        {
            throw new InvalidDataException("SafeTensors graph schema, required arrays or counts are invalid.");
        }

        foreach (var input in graph.Inputs)
        {
            if (input is null || !ValidName(input.Name) || input.Shape is null || input.Shape.Length is 0 or > 2 ||
                input.Shape.Any(dimension => dimension is null || dimension.Minimum <= 0 ||
                    dimension.Maximum < dimension.Minimum || (dimension.Symbol is null ?
                        dimension.Minimum != dimension.Maximum : !ValidName(dimension.Symbol))))
            {
                throw new InvalidDataException("SafeTensors graph input shape or name is invalid.");
            }
        }

        foreach (var node in graph.Nodes)
        {
            if (node is null || !ValidName(node.Name) || !ValidName(node.Operation) || !ValidName(node.Output) ||
                node.Inputs is null || node.Inputs.Length is 0 or > 3 || node.Inputs.Any(name => !ValidName(name)))
            {
                throw new InvalidDataException("SafeTensors graph node structure is invalid.");
            }
        }

        if (graph.Outputs.Any(name => !ValidName(name)))
        {
            throw new InvalidDataException("SafeTensors graph output name is invalid.");
        }
    }

    private static bool ValidName(string? name) => !string.IsNullOrWhiteSpace(name) && name.Length <= 256;
}
