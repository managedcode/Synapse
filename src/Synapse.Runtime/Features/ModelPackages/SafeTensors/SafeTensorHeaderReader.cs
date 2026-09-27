using System.Buffers.Binary;
using System.Text.Json;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;

/// <summary>Reads and validates an untrusted Safetensors header without loading tensor payloads.</summary>
public static class SafeTensorHeaderReader
{
    private const int HeaderPrefixBytes = sizeof(long);
    private const int MaximumHeaderBytes = 16 * 1024 * 1024;
    private const int MaximumTensorCount = 100_000;
    private const int MaximumRank = 8;

    /// <summary>Reads a bounded index and validates every physical range.</summary>
    public static SafeTensorIndex Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(
            Path.GetFullPath(path),
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4096,
            FileOptions.SequentialScan);
        var headerLength = ReadHeaderLength(stream);
        var dataOffset = checked(HeaderPrefixBytes + headerLength);
        if (dataOffset > stream.Length)
        {
            throw new InvalidDataException("Safetensors header extends beyond the source file.");
        }

        var header = new byte[headerLength];
        stream.ReadExactly(header);
        return ParseHeader(header, stream.Length, dataOffset);
    }

    private static int ReadHeaderLength(Stream stream)
    {
        Span<byte> prefix = stackalloc byte[HeaderPrefixBytes];
        stream.ReadExactly(prefix);
        var length = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (length is 0 or > MaximumHeaderBytes)
        {
            throw new InvalidDataException(
                $"Safetensors header length {length} is outside [1, {MaximumHeaderBytes}].");
        }

        return checked((int)length);
    }

    private static SafeTensorIndex ParseHeader(byte[] header, long fileLength, long dataOffset)
    {
        using var document = JsonDocument.Parse(header, new JsonDocumentOptions
        {
            AllowTrailingCommas = false,
            CommentHandling = JsonCommentHandling.Disallow,
            MaxDepth = 16,
        });
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("Safetensors header root must be an object.");
        }

        var names = new HashSet<string>(StringComparer.Ordinal);
        var tensors = new List<SafeTensorInfo>();
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var property in document.RootElement.EnumerateObject())
        {
            if (!names.Add(property.Name))
            {
                throw new InvalidDataException($"Safetensors header repeats key '{property.Name}'.");
            }

            if (property.NameEquals("__metadata__"))
            {
                ReadMetadata(property.Value, metadata);
                continue;
            }

            if (tensors.Count == MaximumTensorCount)
            {
                throw new InvalidDataException($"Safetensors tensor count exceeds {MaximumTensorCount}.");
            }

            tensors.Add(ReadTensor(property, dataOffset, fileLength));
        }

        ValidateRanges(tensors, dataOffset, fileLength);
        return new SafeTensorIndex(fileLength, dataOffset, tensors, metadata);
    }

    private static SafeTensorInfo ReadTensor(
        JsonProperty property,
        long dataOffset,
        long fileLength)
    {
        if (property.Value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Tensor '{property.Name}' descriptor must be an object.");
        }

        EnsureUniqueProperties(property.Name, property.Value);
        var dataType = ParseDataType(GetRequired(property, "dtype").GetString());
        var shape = ReadShape(property.Name, GetRequired(property, "shape"));
        var (start, end) = ReadOffsets(property.Name, GetRequired(property, "data_offsets"));
        var byteLength = checked(end - start);
        var expectedLength = checked(ElementCount(shape) * BytesPerElement(dataType));
        if (byteLength != expectedLength || checked(dataOffset + end) > fileLength)
        {
            throw new InvalidDataException($"Tensor '{property.Name}' has inconsistent shape or byte range.");
        }

        return new SafeTensorInfo(
            property.Name,
            dataType,
            shape,
            checked(dataOffset + start),
            byteLength);
    }

    private static void ReadMetadata(JsonElement element, IDictionary<string, string> destination)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("Safetensors metadata must be an object.");
        }

        foreach (var property in element.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.String ||
                !destination.TryAdd(property.Name, property.Value.GetString()!))
            {
                throw new InvalidDataException("Safetensors metadata must contain unique string values.");
            }
        }
    }

    private static void EnsureUniqueProperties(string tensorName, JsonElement descriptor)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in descriptor.EnumerateObject())
        {
            if (!names.Add(property.Name))
            {
                throw new InvalidDataException($"Tensor '{tensorName}' repeats property '{property.Name}'.");
            }
        }
    }

    private static JsonElement GetRequired(JsonProperty tensor, string propertyName)
    {
        if (!tensor.Value.TryGetProperty(propertyName, out var value))
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' is missing '{propertyName}'.");
        }

        return value;
    }

    private static long[] ReadShape(string tensorName, JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Array || element.GetArrayLength() is 0 or > MaximumRank)
        {
            throw new InvalidDataException($"Tensor '{tensorName}' rank must be inside [1, {MaximumRank}].");
        }

        var shape = new long[element.GetArrayLength()];
        var index = 0;
        foreach (var dimension in element.EnumerateArray())
        {
            if (!dimension.TryGetInt64(out shape[index]) || shape[index] <= 0)
            {
                throw new InvalidDataException($"Tensor '{tensorName}' has a non-positive dimension.");
            }

            index++;
        }

        _ = ElementCount(shape);
        return shape;
    }

    private static (long Start, long End) ReadOffsets(string tensorName, JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Array || element.GetArrayLength() != 2)
        {
            throw new InvalidDataException($"Tensor '{tensorName}' must declare exactly two data offsets.");
        }

        var offsets = element.EnumerateArray().ToArray();
        if (!offsets[0].TryGetInt64(out var start) || !offsets[1].TryGetInt64(out var end) ||
            start < 0 || end <= start)
        {
            throw new InvalidDataException($"Tensor '{tensorName}' has an invalid data range.");
        }

        return (start, end);
    }

    private static void ValidateRanges(
        IEnumerable<SafeTensorInfo> tensors,
        long dataOffset,
        long fileLength)
    {
        var previousEnd = dataOffset;
        foreach (var tensor in tensors.OrderBy(tensor => tensor.Offset))
        {
            if (tensor.Offset < previousEnd || checked(tensor.Offset + tensor.ByteLength) > fileLength)
            {
                throw new InvalidDataException($"Tensor '{tensor.Name}' overlaps or exceeds the source file.");
            }

            previousEnd = checked(tensor.Offset + tensor.ByteLength);
        }
    }

    private static long ElementCount(IEnumerable<long> shape) => shape.Aggregate(
        1L,
        static (count, dimension) => checked(count * dimension));

    private static SafeTensorDataType ParseDataType(string? value) => value switch
    {
        "F32" => SafeTensorDataType.F32,
        "F16" => SafeTensorDataType.F16,
        "BF16" => SafeTensorDataType.BF16,
        "I64" => SafeTensorDataType.I64,
        "I32" => SafeTensorDataType.I32,
        "U8" => SafeTensorDataType.U8,
        "BOOL" => SafeTensorDataType.Boolean,
        _ => throw new InvalidDataException($"Safetensors dtype '{value}' is unsupported."),
    };

    private static int BytesPerElement(SafeTensorDataType dataType) => dataType switch
    {
        SafeTensorDataType.F32 or SafeTensorDataType.I32 => 4,
        SafeTensorDataType.F16 or SafeTensorDataType.BF16 => 2,
        SafeTensorDataType.I64 => 8,
        SafeTensorDataType.U8 or SafeTensorDataType.Boolean => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dataType)),
    };
}
