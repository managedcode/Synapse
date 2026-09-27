using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

internal sealed class CanonicalHashWriter(IncrementalHash hash)
{
    public void WriteBoolean(bool value) => WriteByte(value ? (byte)1 : (byte)0);

    public void WriteSingle(float value) => WriteInt32(BitConverter.SingleToInt32Bits(value));

    public void WriteCount(int value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value);
        WriteInt32(value);
    }

    public void WriteNullableString(string? value)
    {
        WriteBoolean(value is not null);
        if (value is not null)
        {
            WriteString(value);
        }
    }

    public void WriteString(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var bytes = Encoding.UTF8.GetBytes(value);
        WriteCount(bytes.Length);
        hash.AppendData(bytes);
    }

    public void WriteByte(byte value)
    {
        Span<byte> bytes = [value];
        hash.AppendData(bytes);
    }

    public void WriteUInt16(ushort value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ushort)];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
        hash.AppendData(bytes);
    }

    public void WriteInt32(int value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        hash.AppendData(bytes);
    }

    public void WriteUInt32(uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        hash.AppendData(bytes);
    }

    public void WriteInt64(long value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(long)];
        BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
        hash.AppendData(bytes);
    }
}
