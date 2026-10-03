using System.Buffers.Binary;
using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

/// <summary>Length-delimited protobuf reads share one seekable stream and cannot cross their parent range.</summary>
internal sealed class BoundedProtoReader(Stream stream, long end, CancellationToken cancellationToken)
{
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private readonly HashSet<int> _singularFields = [];
    private int _fields;

    public bool HasData => stream.Position < end;
    public long Remaining => end - stream.Position;

    public (int Field, int Wire) Next()
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (++_fields > 100_000)
        {
            throw Error("Too many fields in an ONNX protobuf message.");
        }
        var key = Varint();
        if (key == 0 || key > uint.MaxValue || (key & 7) is not (0 or 1 or 2 or 5))
        {
            throw Error("Invalid ONNX protobuf field key or wire type.");
        }
        return ((int)(key >> 3), (int)(key & 7));
    }

    public void Once(int field)
    {
        if (!_singularFields.Add(field))
        {
            throw Error($"Duplicate ONNX protobuf field {field}.");
        }
    }

    public long Integer(int wire)
    {
        RequireWire(wire, 0);
        return unchecked((long)Varint());
    }

    public float Float(int wire)
    {
        RequireWire(wire, 5);
        Span<byte> bytes = stackalloc byte[4];
        Read(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    public string Text(int wire)
    {
        var child = Message(wire);
        if (child.Remaining > 4096)
        {
            throw Error("ONNX names and symbols must fit 4 KiB.");
        }
        var bytes = new byte[(int)child.Remaining];
        child.Read(bytes);
        try
        {
            return StrictUtf8.GetString(bytes);
        }
        catch (DecoderFallbackException exception)
        {
            throw new InvalidDataException("Malformed ONNX UTF-8 string.", exception);
        }
    }

    public BoundedProtoReader Message(int wire)
    {
        RequireWire(wire, 2);
        var length = Varint();
        if (length > (ulong)Remaining)
        {
            throw Error("ONNX protobuf length exceeds its enclosing message.");
        }
        return new(stream, stream.Position + (long)length, cancellationToken);
    }

    public void Skip(int wire)
    {
        if (wire == 0)
        {
            _ = Varint();
            return;
        }
        var length = wire switch
        {
            1 => 8L,
            5 => 4L,
            2 => Message(wire).Remaining,
            _ => throw Error("Unsupported ONNX protobuf wire type."),
        };
        if (length > Remaining)
        {
            throw Error("Truncated ONNX protobuf field.");
        }
        _ = stream.Seek(length, SeekOrigin.Current);
    }

    public static void RequireWire(int actual, int expected)
    {
        if (actual != expected)
        {
            throw Error("Unexpected ONNX protobuf wire type.");
        }
    }

    public static InvalidDataException Error(string message) => new(message);

    private ulong Varint()
    {
        ulong result = 0;
        Span<byte> next = stackalloc byte[1];
        for (var shift = 0; shift < 70; shift += 7)
        {
            Read(next);
            var value = next[0];
            if (shift == 63 && value > 1)
            {
                throw Error("ONNX protobuf varint overflows 64 bits.");
            }
            result |= (ulong)(value & 127) << shift;
            if (value < 128)
            {
                return result;
            }
        }
        throw Error("Unterminated ONNX protobuf varint.");
    }

    private void Read(Span<byte> bytes)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (bytes.Length > Remaining)
        {
            throw Error("Truncated ONNX protobuf data.");
        }
        stream.ReadExactly(bytes);
    }
}
