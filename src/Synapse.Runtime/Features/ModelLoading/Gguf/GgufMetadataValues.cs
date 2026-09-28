using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

/// <summary>GGUF metadata value primitives: bounded strings and counts, scalar values, and array skipping.</summary>
internal static class GgufMetadataValues
{
    internal static object? ReadValue(BinaryReader reader, uint type, int depth)
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
        SkipArrayElements(reader, elementType, count, depth);
        return null;
    }

    internal static void SkipArrayElements(BinaryReader reader, uint elementType, int count, int depth)
    {
        if (elementType != 8 && TryGetFixedSize(elementType, out var elementSize))
        {
            _ = reader.BaseStream.Seek(checked((long)count * elementSize), SeekOrigin.Current);
            return;
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

    internal static string ReadString(BinaryReader reader)
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

    internal static int ReadBoundedCount(BinaryReader reader, string name, int maximum)
    {
        var count = reader.ReadUInt64();
        return count <= (ulong)maximum
            ? checked((int)count)
            : throw new InvalidDataException($"GGUF {name} count {count} exceeds {maximum}.");
    }
}
