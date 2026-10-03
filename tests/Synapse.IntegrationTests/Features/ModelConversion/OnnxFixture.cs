using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

/// <summary>Real ONNX protobuf files, encoded independently of the production reader.</summary>
internal sealed class OnnxFixture : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"synapse-onnx-{Guid.NewGuid():N}");

    public string Write(byte[] model)
    {
        Directory.CreateDirectory(_directory);
        var path = Path.Combine(_directory, $"{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, model);
        return path;
    }

    public static byte[] Model(byte[] graph, long opset = 13, params byte[][] extras) => Join(
        Integer(1, 9), Message(7, graph), Message(8, Integer(2, opset)), Join(extras));

    public static byte[] Graph(byte[][] nodes, byte[][] tensors, byte[][] inputs, byte[][] outputs, params byte[][] extras) => Join(
        Join([.. nodes.Select(x => Message(1, x))]), Text(2, "fixture"),
        Join([.. tensors.Select(x => Message(5, x))]), Join([.. inputs.Select(x => Message(11, x))]),
        Join([.. outputs.Select(x => Message(12, x))]), Join(extras));

    public static byte[] Value(string name, params object[] shape) => Join(Text(1, name), Message(2,
        Message(1, Join(Integer(1, 1), Message(2, Join([.. shape.Select(d => Message(1,
            d is string symbol ? Text(2, symbol) : Integer(1, Convert.ToInt64(d, CultureInfo.InvariantCulture))))]))))));

    public static byte[] Tensor(string name, long[] shape, float[] values, bool raw = true, params byte[][] extras)
    {
        var data = new byte[values.Length * sizeof(float)];
        for (var i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(i * sizeof(float)), values[i]);
        }
        return Join(Join([.. shape.Select(x => Integer(1, x))]), Integer(2, 1), Text(8, name),
            Message(raw ? 9 : 4, data), Join(extras));
    }

    public static byte[] Node(string op, string[] inputs, string output = "y", params byte[][] attributes) => Join(
        Join([.. inputs.Select(x => Text(1, x))]), Text(2, output), Text(4, op),
        Join([.. attributes.Select(x => Message(5, x))]));

    public static byte[] IntAttribute(string name, long value) => Join(Text(1, name), Integer(3, value), Integer(20, 2));

    public static byte[] FloatAttribute(string name, float value)
    {
        var data = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(data, value);
        return Join(Text(1, name), Varint((2 << 3) | 5), data, Integer(20, 1));
    }

    public static byte[] Integer(int field, long value) => Join(Varint((ulong)(field << 3)), Varint(unchecked((ulong)value)));
    public static byte[] Text(int field, string value) => Message(field, Encoding.UTF8.GetBytes(value));
    public static byte[] Message(int field, byte[] content) => Join(Varint((ulong)((field << 3) | 2)), Varint((ulong)content.Length), content);
    public static byte[] Join(params byte[][] parts) => [.. parts.SelectMany(x => x)];

    private static byte[] Varint(ulong value)
    {
        var result = new List<byte>();
        do
        {
            var next = (byte)(value & 127);
            value >>= 7;
            result.Add(value == 0 ? next : (byte)(next | 128));
        } while (value != 0);
        return [.. result];
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, true);
        }
    }
}
