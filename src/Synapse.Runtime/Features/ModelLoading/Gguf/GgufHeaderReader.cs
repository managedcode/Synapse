using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal static class GgufHeaderReader
{
    private const uint Magic = 0x4655_4747;
    private const uint Float32Type = 0;
    private const uint Q8Type = 8;
    private const int DefaultAlignment = 32;

    public static GgufDescriptor Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        ValidateHeader(reader);
        var tensorCount = ReadBoundedCount(reader, "tensor", 1_000_000);
        var metadataCount = ReadBoundedCount(reader, "metadata", 1_000_000);
        var metadata = ReadMetadata(reader, metadataCount);
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
            MaterializeTensorInfos(tensorHeaders, dataOffset, stream.Length));
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

    private static Dictionary<string, object> ReadMetadata(BinaryReader reader, int count)
    {
        var metadata = new Dictionary<string, object>(count, StringComparer.Ordinal);
        for (var index = 0; index < count; index++)
        {
            var key = ReadString(reader);
            var type = reader.ReadUInt32();
            var capture = type != 9;
            var value = ReadValue(reader, type, depth: 0);
            if (capture && value is not null)
            {
                metadata.Add(key, value);
            }
        }

        return metadata;
    }

    private static List<TensorHeader> ReadTensorHeaders(BinaryReader reader, int count)
    {
        var tensors = new List<TensorHeader>(count);
        for (var index = 0; index < count; index++)
        {
            var name = ReadString(reader);
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

    private static object? ReadValue(BinaryReader reader, uint type, int depth)
    {
        if (depth > 2)
        {
            throw new InvalidDataException("Nested GGUF metadata is too deep.");
        }

        return type switch
        {
            0 => reader.ReadByte(),
            1 => reader.ReadSByte(),
            2 => reader.ReadUInt16(),
            3 => reader.ReadInt16(),
            4 => reader.ReadUInt32(),
            5 => reader.ReadInt32(),
            6 => reader.ReadSingle(),
            7 => reader.ReadByte() != 0,
            8 => ReadString(reader),
            9 => ReadArray(reader, depth + 1),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new InvalidDataException($"GGUF metadata type {type} is unknown."),
        };
    }

    private static object? ReadArray(BinaryReader reader, int depth)
    {
        var elementType = reader.ReadUInt32();
        var count = ReadBoundedCount(reader, "array element", 10_000_000);
        if (elementType != 8 && TryGetFixedSize(elementType, out var elementSize))
        {
            _ = reader.BaseStream.Seek(checked((long)count * elementSize), SeekOrigin.Current);
            return null;
        }

        for (var index = 0; index < count; index++)
        {
            if (elementType == 8)
            {
                SkipString(reader);
            }
            else
            {
                _ = ReadValue(reader, elementType, depth);
            }
        }

        return null;
    }

    /// <summary>Validates and skips an unused metadata string without materializing it.</summary>
    private static void SkipString(BinaryReader reader)
    {
        var length = reader.ReadUInt64();
        if (length > 16 * 1024 * 1024)
        {
            throw new InvalidDataException($"GGUF string length {length} exceeds the safety limit.");
        }

        // ReadExactly throws EndOfStreamException for a truncated file; querying Length per string would
        // cost one fstat call for each of the ~300k tokenizer entries.
        var stream = reader.BaseStream;
        Span<byte> discard = stackalloc byte[256];
        for (var remaining = (int)length; remaining > 0; remaining -= discard.Length)
        {
            stream.ReadExactly(discard[..Math.Min(remaining, discard.Length)]);
        }
    }

    private static bool TryGetFixedSize(uint type, out int size)
    {
        size = type switch
        {
            0 or 1 or 7 => 1,
            2 or 3 => 2,
            4 or 5 or 6 => 4,
            10 or 11 or 12 => 8,
            _ => 0,
        };
        return size != 0;
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = reader.ReadUInt64();
        if (length > 16 * 1024 * 1024)
        {
            throw new InvalidDataException($"GGUF string length {length} exceeds the safety limit.");
        }

        var bytes = reader.ReadBytes(checked((int)length));
        if ((ulong)bytes.Length != length)
        {
            throw new EndOfStreamException("GGUF string is truncated.");
        }

        return Encoding.UTF8.GetString(bytes);
    }

    private static int ReadBoundedCount(BinaryReader reader, string name, int maximum)
    {
        var count = reader.ReadUInt64();
        return count <= (ulong)maximum
            ? checked((int)count)
            : throw new InvalidDataException($"GGUF {name} count {count} exceeds {maximum}.");
    }

    private sealed record TensorHeader(
        string Name,
        ulong[] Dimensions,
        uint Type,
        ulong RelativeOffset);
}

internal sealed record GgufDescriptor(
    IReadOnlyDictionary<string, object> Metadata,
    IReadOnlyDictionary<string, GgufTensorInfo> Tensors);
