using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

internal sealed class SafeTensorConversionFixture : IDisposable
{
    private readonly string directory = Path.Combine(Path.GetTempPath(), $"synapse-safetensor-{Guid.NewGuid():N}");

    public SafeTensorConversionFixture()
    {
        Directory.CreateDirectory(directory);
        Source = Path.Combine(directory, "weights.safetensors");
        Graph = Path.Combine(directory, "graph.json");
        File.WriteAllText(Graph, DefaultGraph, Encoding.UTF8);
    }

    public string Source { get; }

    public string Graph { get; }

    public const string DefaultGraph = /*lang=json,strict*/ """
        {"schema_version":1,"inputs":[{"name":"x","shape":[{"symbol":null,"minimum":2,"maximum":2}]}],
        "outputs":["y"],"nodes":[{"name":"linear","operation":"Linear","inputs":["x","weight"],"output":"y"}]}
        """;

    public void Write(params SafeTensorConversionTensor[] tensors)
    {
        using var header = new MemoryStream();
        long length = 0;
        using (var writer = new Utf8JsonWriter(header))
        {
            writer.WriteStartObject();
            foreach (var tensor in tensors)
            {
                writer.WritePropertyName(tensor.Name);
                writer.WriteStartObject();
                writer.WriteString("dtype", tensor.Dtype);
                writer.WriteStartArray("shape");
                foreach (var dimension in tensor.Shape)
                {
                    writer.WriteNumberValue(dimension);
                }
                writer.WriteEndArray();
                writer.WriteStartArray("data_offsets");
                writer.WriteNumberValue(length);
                length = checked(length + TensorBytes(tensor));
                writer.WriteNumberValue(length);
                writer.WriteEndArray();
                writer.WriteEndObject();
            }

            writer.WriteEndObject();
        }

        using var source = File.Create(Source);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)header.Length);
        source.Write(prefix);
        source.Write(header.ToArray());
        var payloadStart = source.Position;
        source.SetLength(checked(payloadStart + length));
        foreach (var tensor in tensors)
        {
            source.Write(tensor.Bytes);
            source.Position += TensorBytes(tensor) - tensor.Bytes.Length;
        }
    }

    public static byte[] Singles(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (var index = 0; index < values.Length; index++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(index * 4), values[index]);
        }
        return bytes;
    }

    public static byte[] Halves(params ushort[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (var index = 0; index < values.Length; index++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2), values[index]);
        }
        return bytes;
    }

    public void Dispose() => Directory.Delete(directory, true);

    private static long TensorBytes(SafeTensorConversionTensor tensor) =>
        checked(tensor.Shape.Aggregate(1L, (count, value) => checked(count * value)) * tensor.BytesPerElement);
}

internal sealed record SafeTensorConversionTensor(string Name, string Dtype, long[] Shape, int BytesPerElement, byte[] Bytes);
