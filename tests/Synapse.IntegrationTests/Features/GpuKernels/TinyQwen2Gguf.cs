using System.Text;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>
/// Writes a deterministic tiny Qwen2 GGUF (v3): Q8_0 matrices, FP32 norms and biases, seeded values. It runs real
/// math at shapes the pinned 0.5B model does not have (for example head dimension 128, as in Qwen2.5-7B), so every
/// backend can be compared without a large download.
/// </summary>
internal static class TinyQwen2Gguf
{
    private const uint Float32Type = 0;
    private const uint Q8ZeroType = 8;
    private const int Alignment = 32;

    public sealed record Shape(int Layers, int Heads, int KeyValueHeads, int HeadDimension, int FeedForward, int Vocabulary, int Context)
    {
        public int Hidden => Heads * HeadDimension;

        public int KeyValueWidth => KeyValueHeads * HeadDimension;
    }

    /// <summary>
    /// Writes the model to a new temporary file and returns its path; the caller deletes it. With
    /// <paramref name="keptLayers"/>, every layer is still drawn from the seeded stream but only the kept ones are
    /// written, renumbered in order: the shallower model a layer drop (ADR-019) must equal.
    /// </summary>
    /// <param name="shape">Model dimensions.</param>
    /// <param name="seed">Seed of every drawn value.</param>
    /// <param name="keptLayers">Layers to write, or null for all.</param>
    /// <param name="encodings">Matrix encoding by tensor name (Q8_0 when null); norms and biases stay F32.</param>
    /// <param name="dequantized">Writes each K-quant matrix as F32 holding its exact decoded values: the oracle model.</param>
    public static string Write(
        Shape shape,
        int seed,
        IReadOnlyList<int>? keptLayers = null,
        Func<string, TinyEncoding>? encodings = null,
        bool dequantized = false)
    {
        var random = new Random(seed);
        (string, ulong[], uint, byte[]) Matrix(string name, int columns, int rows, Random source, float scale = 0)
        {
            return EncodeMatrix(name, columns, rows, source, scale, encodings?.Invoke(name) ?? TinyEncoding.Q8Zero, dequantized);
        }

        var tensors = new List<(string Name, ulong[] Dimensions, uint Type, byte[] Data)>
        {
            Matrix("token_embd.weight", shape.Hidden, shape.Vocabulary, random, scale: 1.0f),
        };
        var written = 0;
        for (var layer = 0; layer < shape.Layers; layer++)
        {
            var kept = keptLayers is null || keptLayers.Contains(layer);
            var prefix = kept ? $"blk.{written++}" : "dropped";
            var first = tensors.Count;
            tensors.Add(Vector($"{prefix}.attn_norm.weight", shape.Hidden, random, center: 1.0f, scale: 0.2f));
            tensors.Add(Matrix($"{prefix}.attn_q.weight", shape.Hidden, shape.Hidden, random));
            tensors.Add(Matrix($"{prefix}.attn_k.weight", shape.Hidden, shape.KeyValueWidth, random));
            tensors.Add(Matrix($"{prefix}.attn_v.weight", shape.Hidden, shape.KeyValueWidth, random));
            tensors.Add(Vector($"{prefix}.attn_q.bias", shape.Hidden, random, center: 0.0f, scale: 0.5f));
            tensors.Add(Vector($"{prefix}.attn_k.bias", shape.KeyValueWidth, random, center: 0.0f, scale: 0.5f));
            tensors.Add(Vector($"{prefix}.attn_v.bias", shape.KeyValueWidth, random, center: 0.0f, scale: 0.1f));
            tensors.Add(Matrix($"{prefix}.attn_output.weight", shape.Hidden, shape.Hidden, random));
            tensors.Add(Vector($"{prefix}.ffn_norm.weight", shape.Hidden, random, center: 1.0f, scale: 0.2f));
            tensors.Add(Matrix($"{prefix}.ffn_gate.weight", shape.Hidden, shape.FeedForward, random));
            tensors.Add(Matrix($"{prefix}.ffn_up.weight", shape.Hidden, shape.FeedForward, random));
            tensors.Add(Matrix($"{prefix}.ffn_down.weight", shape.FeedForward, shape.Hidden, random));
            if (!kept)
            {
                tensors.RemoveRange(first, tensors.Count - first);
            }
        }

        tensors.Add(Vector("output_norm.weight", shape.Hidden, random, center: 1.0f, scale: 0.2f));
        tensors.Add(Matrix("output.weight", shape.Hidden, shape.Vocabulary, random));
        var path = Path.Combine(Path.GetTempPath(), $"synapse-tiny-qwen2-{Guid.NewGuid():N}.gguf");
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        WriteHeader(writer, shape with { Layers = written }, tensors);
        foreach (var tensor in tensors)
        {
            writer.Write(tensor.Data);
            Pad(writer);
        }

        return path;
    }

    private static void WriteHeader(
        BinaryWriter writer,
        Shape shape,
        List<(string Name, ulong[] Dimensions, uint Type, byte[] Data)> tensors)
    {
        writer.Write("GGUF"u8);
        writer.Write(3u);
        writer.Write((ulong)tensors.Count);
        writer.Write(9ul);
        WriteString(writer, "general.architecture");
        writer.Write(8u);
        WriteString(writer, "qwen2");
        foreach (var (key, value) in new (string, int)[]
        {
            ("qwen2.block_count", shape.Layers),
            ("qwen2.context_length", shape.Context),
            ("qwen2.embedding_length", shape.Hidden),
            ("qwen2.feed_forward_length", shape.FeedForward),
            ("qwen2.attention.head_count", shape.Heads),
            ("qwen2.attention.head_count_kv", shape.KeyValueHeads),
        })
        {
            WriteString(writer, key);
            writer.Write(4u);
            writer.Write((uint)value);
        }

        foreach (var (key, value) in new (string, float)[] { ("qwen2.rope.freq_base", 1_000_000f), ("qwen2.attention.layer_norm_rms_epsilon", 1e-6f) })
        {
            WriteString(writer, key);
            writer.Write(6u);
            writer.Write(value);
        }

        ulong offset = 0;
        foreach (var (name, dimensions, type, data) in tensors)
        {
            WriteString(writer, name);
            writer.Write((uint)dimensions.Length);
            foreach (var dimension in dimensions)
            {
                writer.Write(dimension);
            }

            writer.Write(type);
            writer.Write(offset);
            offset += (ulong)AlignUp(data.Length);
        }

        Pad(writer);
    }

    /// <summary>A GGUF matrix of <paramref name="rows"/> rows of <paramref name="columns"/> values, stored as Q8_0.</summary>
    private static (string, ulong[], uint, byte[]) EncodeMatrix(
        string name,
        int columns,
        int rows,
        Random random,
        float scale,
        TinyEncoding encoding,
        bool dequantized)
    {
        var range = scale > 0 ? scale : 1.5f / MathF.Sqrt(columns);
        var values = new float[columns * rows];
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = (((float)random.NextDouble() * 2) - 1) * range;
        }

        ulong[] dimensions = [(ulong)columns, (ulong)rows];
        if (encoding == TinyEncoding.Q8Zero)
        {
            return (name, dimensions, Q8ZeroType, QuantizeQ8Zero(values));
        }

        var (type, bytes) = encoding == TinyEncoding.Q4K ? (12u, TinyKQuantEncoder.Q4K(values)) : (14u, TinyKQuantEncoder.Q6K(values));
        if (!dequantized)
        {
            return (name, dimensions, type, bytes);
        }

        var decoded = new float[values.Length];
        SourceEncodings.GetGguf(type).Decode(bytes, decoded);
        var data = new byte[decoded.Length * sizeof(float)];
        Buffer.BlockCopy(decoded, 0, data, 0, data.Length);
        return (name, dimensions, Float32Type, data);
    }

    private static (string, ulong[], uint, byte[]) Vector(string name, int length, Random random, float center, float scale)
    {
        var data = new byte[length * sizeof(float)];
        for (var index = 0; index < length; index++)
        {
            BitConverter.TryWriteBytes(data.AsSpan(index * sizeof(float)), center + ((((float)random.NextDouble() * 2) - 1) * scale));
        }

        return (name, [(ulong)length], Float32Type, data);
    }

    /// <summary>ggml Q8_0: per 32 values an FP16 scale <c>max|x| / 127</c> and 32 rounded signed bytes.</summary>
    private static byte[] QuantizeQ8Zero(float[] values)
    {
        var blocks = values.Length / 32;
        var data = new byte[blocks * 34];
        for (var block = 0; block < blocks; block++)
        {
            var slice = values.AsSpan(block * 32, 32);
            var maximum = 0f;
            foreach (var value in slice)
            {
                maximum = Math.Max(maximum, Math.Abs(value));
            }

            var scale = maximum / 127f;
            var inverse = scale == 0 ? 0 : 1 / scale;
            BitConverter.TryWriteBytes(data.AsSpan(block * 34), (Half)scale);
            for (var index = 0; index < 32; index++)
            {
                data[(block * 34) + 2 + index] = unchecked((byte)(sbyte)Math.Round(slice[index] * inverse));
            }
        }

        return data;
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static int AlignUp(int length) => (length + Alignment - 1) / Alignment * Alignment;

    private static void Pad(BinaryWriter writer)
    {
        var position = writer.BaseStream.Position;
        var padding = (Alignment - (int)(position % Alignment)) % Alignment;
        writer.Write(new byte[padding]);
    }
}

/// <summary>How a generated matrix is stored.</summary>
internal enum TinyEncoding
{
    Q8Zero,
    Q4K,
    Q6K,
}
