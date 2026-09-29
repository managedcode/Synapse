using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal static class GgufHeaderReader
{
    private const uint Magic = 0x4655_4747;
    private const uint Float32Type = 0;
    private const uint Q8Type = 8;
    private const uint Q4KType = 12;
    private const uint Q6KType = 14;
    private const int DefaultAlignment = 32;

    public static GgufDescriptor Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        ValidateHeader(reader);
        var tensorCount = GgufMetadataValues.ReadBoundedCount(reader, "tensor", 1_000_000);
        var metadataCount = GgufMetadataValues.ReadBoundedCount(reader, "metadata", 1_000_000);
        var (metadata, arrays) = ReadMetadata(reader, metadataCount);
        var tensorHeaders = ReadTensorHeaders(reader, tensorCount);
        var alignment = metadata.TryGetValue("general.alignment", out var configuredAlignment)
            ? Convert.ToInt32(configuredAlignment, System.Globalization.CultureInfo.InvariantCulture)
            : DefaultAlignment;
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
        {
            throw new InvalidDataException($"GGUF alignment {alignment} is invalid.");
        }

        var dataOffset = checked((stream.Position + alignment - 1) / alignment * alignment);
        return new GgufDescriptor(
            metadata,
            MaterializeTensorInfos(tensorHeaders, dataOffset, stream.Length),
            arrays);
    }

    private static void ValidateHeader(BinaryReader reader)
    {
        if (reader.ReadUInt32() != Magic)
        {
            throw new InvalidDataException("The file is not a GGUF model.");
        }

        var version = reader.ReadUInt32();
        if (version is < 2 or > 3)
        {
            throw new InvalidDataException($"GGUF version {version} is not supported.");
        }
    }

    /// <summary>
    /// Scalars are materialized; arrays are skipped and only located (element type, count, offset of the first
    /// element), so the tokenizer can parse them later from the memory map without a second file read.
    /// </summary>
    private static (Dictionary<string, object> Metadata, Dictionary<string, GgufArrayReference> Arrays) ReadMetadata(
        BinaryReader reader,
        int count)
    {
        var metadata = new Dictionary<string, object>(count, StringComparer.Ordinal);
        var arrays = new Dictionary<string, GgufArrayReference>(StringComparer.Ordinal);
        for (var index = 0; index < count; index++)
        {
            var key = GgufMetadataValues.ReadString(reader);
            var type = reader.ReadUInt32();
            if (type == 9)
            {
                var elementType = reader.ReadUInt32();
                var elements = GgufMetadataValues.ReadBoundedCount(reader, "array element", 10_000_000);
                arrays.Add(key, new GgufArrayReference(elementType, elements, reader.BaseStream.Position));
                GgufMetadataValues.SkipArrayElements(reader, elementType, elements, depth: 1);
                continue;
            }

            if (GgufMetadataValues.ReadValue(reader, type, depth: 0) is { } value)
            {
                metadata.Add(key, value);
            }
        }

        return (metadata, arrays);
    }

    private static List<TensorHeader> ReadTensorHeaders(BinaryReader reader, int count)
    {
        var tensors = new List<TensorHeader>(count);
        for (var index = 0; index < count; index++)
        {
            var name = GgufMetadataValues.ReadString(reader);
            var dimensionCount = reader.ReadUInt32();
            if (dimensionCount is 0 or > 8)
            {
                throw new InvalidDataException($"Tensor '{name}' has invalid rank {dimensionCount}.");
            }

            var dimensions = new ulong[dimensionCount];
            for (var dimension = 0; dimension < dimensions.Length; dimension++)
            {
                dimensions[dimension] = reader.ReadUInt64();
            }

            tensors.Add(new TensorHeader(name, dimensions, reader.ReadUInt32(), reader.ReadUInt64()));
        }

        return tensors;
    }

    private static Dictionary<string, GgufTensorInfo> MaterializeTensorInfos(
        IReadOnlyList<TensorHeader> headers,
        long dataOffset,
        long fileLength)
    {
        var tensors = new Dictionary<string, GgufTensorInfo>(headers.Count, StringComparer.Ordinal);
        foreach (var header in headers)
        {
            var elementCount = header.Dimensions.Aggregate(1L, static (count, dimension) =>
                checked(count * checked((long)dimension)));
            var byteLength = header.Type switch
            {
                Float32Type => checked(elementCount * sizeof(float)),
                Q8Type when elementCount % 32 == 0 => checked(elementCount / 32 * 34),
                Q4KType when elementCount % 256 == 0 => checked(elementCount / 256 * 144),
                Q6KType when elementCount % 256 == 0 => checked(elementCount / 256 * 210),
                _ => throw new NotSupportedException(
                    $"Tensor '{header.Name}' uses unsupported GGML type {header.Type}."),
            };
            var absoluteOffset = checked(dataOffset + checked((long)header.RelativeOffset));
            if (absoluteOffset < dataOffset || absoluteOffset + byteLength > fileLength)
            {
                throw new InvalidDataException($"Tensor '{header.Name}' is outside the GGUF file.");
            }

            tensors.Add(
                header.Name,
                new GgufTensorInfo(
                    header.Name,
                    header.Dimensions,
                    header.Type,
                    absoluteOffset,
                    byteLength));
        }

        return tensors;
    }

    private sealed record TensorHeader(
        string Name,
        ulong[] Dimensions,
        uint Type,
        ulong RelativeOffset);
}

internal sealed record GgufDescriptor(
    IReadOnlyDictionary<string, object> Metadata,
    IReadOnlyDictionary<string, GgufTensorInfo> Tensors,
    IReadOnlyDictionary<string, GgufArrayReference> Arrays);

/// <summary>A metadata array located in the file: element type, element count, and offset of the first element.</summary>
internal sealed record GgufArrayReference(uint ElementType, int Count, long Offset);
