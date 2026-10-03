using ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

/// <summary>Normalizes bounded SafeTensors weights with an explicit inert graph sidecar.</summary>
public static class SafeTensorsModelImporter
{
    private const long MaximumSourceBytes = 128L * 1024 * 1024;
    private const int MaximumTensorCount = 1024;
    private const long MaximumTensorElements = 1_000_000;
    private const long MaximumTotalElements = 16_000_000;

    /// <summary>Reads finite FP32/F16/BF16 weights without execution or implicit architecture inference.</summary>
    public static ConversionModel Import(string sourcePath, string graphPath, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(graphPath);
        if (new FileInfo(sourcePath).Length > MaximumSourceBytes)
        {
            throw new InvalidDataException($"SafeTensors source exceeds {MaximumSourceBytes} bytes.");
        }

        var index = SafeTensorHeaderReader.Read(sourcePath);
        ValidateIndex(index);
        var graph = SafeTensorGraphReader.Read(graphPath, cancellationToken);
        ValidateRequiredTensors(graph, index);
        using var stream = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length != index.FileLength)
        {
            throw new InvalidDataException("SafeTensors source changed after header validation.");
        }

        var tensors = new ConversionTensor[index.Tensors.Count];
        for (var indexNumber = 0; indexNumber < tensors.Length; indexNumber++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            tensors[indexNumber] = Decode(stream, index.Tensors[indexNumber], cancellationToken);
        }

        cancellationToken.ThrowIfCancellationRequested();
        return new ConversionModel("safetensors", graph, tensors);
    }

    private static void ValidateIndex(SafeTensorIndex index)
    {
        if (index.FileLength > MaximumSourceBytes || index.Tensors.Count is 0 or > MaximumTensorCount)
        {
            throw new InvalidDataException("SafeTensors source size or tensor count exceeds conversion bounds.");
        }

        long totalElements = 0;
        foreach (var tensor in index.Tensors)
        {
            if (string.IsNullOrWhiteSpace(tensor.Name) || tensor.Name.Length > 256 ||
                tensor.ElementCount > MaximumTensorElements ||
                (totalElements += tensor.ElementCount) > MaximumTotalElements)
            {
                throw new InvalidDataException($"SafeTensors tensor '{tensor.Name}' exceeds conversion bounds.");
            }

            _ = Decoder(tensor.DataType);
        }
    }

    private static void ValidateRequiredTensors(ConversionGraph graph, SafeTensorIndex index)
    {
        var values = graph.Inputs.Select(input => input.Name).Concat(graph.Nodes.Select(node => node.Output))
            .ToHashSet(StringComparer.Ordinal);
        var tensors = index.Tensors.Select(tensor => tensor.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var name in graph.Nodes.SelectMany(node => node.Inputs).Concat(graph.Outputs))
        {
            if (!values.Contains(name) && !tensors.Contains(name))
            {
                throw new InvalidDataException($"SafeTensors graph requires missing tensor or value '{name}'.");
            }
        }
    }

    private static ConversionTensor Decode(FileStream stream, SafeTensorInfo tensor, CancellationToken cancellationToken)
    {
        var bytes = new byte[checked((int)tensor.ByteLength)];
        stream.Position = tensor.Offset;
        for (var offset = 0; offset < bytes.Length; offset += 64 * 1024)
        {
            cancellationToken.ThrowIfCancellationRequested();
            stream.ReadExactly(bytes.AsSpan(offset, Math.Min(64 * 1024, bytes.Length - offset)));
        }

        var data = new float[checked((int)tensor.ElementCount)];
        Decoder(tensor.DataType).Decode(bytes, data);
        cancellationToken.ThrowIfCancellationRequested();
        return new ConversionTensor(tensor.Name, [.. tensor.Shape], data);
    }

    private static ISourceTensorDecoder Decoder(SafeTensorDataType dataType) => dataType switch
    {
        SafeTensorDataType.F32 => SourceEncodings.Fp32,
        SafeTensorDataType.F16 => SourceEncodings.Fp16,
        SafeTensorDataType.BF16 => SourceEncodings.Bf16,
        SafeTensorDataType.I32 or SafeTensorDataType.I64 or SafeTensorDataType.U8 or SafeTensorDataType.Boolean =>
            throw new NotSupportedException($"SafeTensors conversion requires F32, F16 or BF16; '{dataType}' is unsupported."),
        _ => throw new NotSupportedException($"SafeTensors conversion requires F32, F16 or BF16; '{dataType}' is unsupported."),
    };
}
