using System.Text;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

internal sealed class CompiledPackageFixture : IDisposable
{
    private static readonly string[] ArrayTokens = ["<unk>", "a", "б"];
    private static readonly (string Key, string Value)[] TokenizerMetadata =
        [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen2")];

    internal CompiledPackageFixture(bool arrays = false, TinyEncoding encoding = TinyEncoding.Q8Zero, bool tokenizer = false)
    {
        Source = TinyQwen2Gguf.Write(new(1, 1, 1, 256, 256, 64, 32), seed: 17, encodings: _ => encoding);
        Destination = Path.ChangeExtension(Source, ".synapse");
        if (tokenizer)
        {
            AddTokenizer(Source);
        }
        else if (arrays)
        {
            AddArrays(Source);
        }
    }

    internal string Source { get; }
    internal string Destination { get; }

    public void Dispose()
    {
        File.Delete(Source);
        File.Delete(Destination);
        foreach (var partial in Directory.EnumerateFiles(Path.GetDirectoryName(Destination)!, Path.GetFileName(Destination) + ".*.partial.synapse"))
        {
            File.Delete(partial);
        }
    }

    private static void RewriteMetadata(string path, int addedMetadata, Action<BinaryWriter> add)
    {
        var descriptor = GgufHeaderReader.Read(path);
        var bytes = File.ReadAllBytes(path);
        using var input = new MemoryStream(bytes);
        using var reader = new BinaryReader(input, Encoding.UTF8, leaveOpen: true);
        input.Position = 24;
        for (var index = 0; index < 9; index++)
        {
            _ = GgufMetadataValues.ReadString(reader);
            _ = GgufMetadataValues.ReadValue(reader, reader.ReadUInt32(), 0);
        }

        var metadataEnd = checked((int)input.Position);
        using var output = new MemoryStream();
        using var writer = new BinaryWriter(output, Encoding.UTF8, leaveOpen: true);
        writer.Write(bytes.AsSpan(0, metadataEnd));
        add(writer);
        AddAlignmentMetadata(writer, metadataEnd);
        writer.Write(bytes.AsSpan(metadataEnd, checked((int)descriptor.DataOffset) - metadataEnd));
        while (output.Length % 32 != 0)
        {
            writer.Write((byte)0);
        }

        writer.Write(bytes.AsSpan(checked((int)descriptor.DataOffset)));
        output.Position = 16;
        writer.Write((ulong)(10 + addedMetadata));
        File.WriteAllBytes(path, output.ToArray());
    }

    private static void AddArrays(string path) => RewriteMetadata(path, 2, static writer =>
    {
        Array(writer, "tokenizer.ggml.tokens", 8, 3);
        foreach (var token in ArrayTokens)
        {
            String(writer, token);
        }

        Array(writer, "tokenizer.ggml.token_type", 5, 3);
        writer.Write(2);
        writer.Write(1);
        writer.Write(1);
    });

    private static void AddTokenizer(string path) => RewriteMetadata(path, 5, static writer =>
    {
        foreach (var (key, value) in TokenizerMetadata)
        {
            String(writer, key);
            writer.Write(8u);
            String(writer, value);
        }

        Array(writer, "tokenizer.ggml.tokens", 8, 64);
        String(writer, "a");
        String(writer, "b");
        String(writer, "ab");
        for (var index = 3; index < 64; index++)
        {
            String(writer, "token" + index.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        Array(writer, "tokenizer.ggml.token_type", 5, 64);
        for (var index = 0; index < 64; index++)
        {
            writer.Write(1);
        }

        Array(writer, "tokenizer.ggml.merges", 8, 1);
        String(writer, "a b");
    });

    private static void Array(BinaryWriter writer, string key, uint elementType, int count)
    {
        String(writer, key);
        writer.Write(9u);
        writer.Write(elementType);
        writer.Write((ulong)count);
    }

    private static void String(BinaryWriter writer, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void AddAlignmentMetadata(BinaryWriter writer, long metadataEnd)
    {
        const string Key = "synapse.test.padding";
        var extra = 8 + Key.Length + 4 + 8;
        var needed = (32 - (int)((writer.BaseStream.Position - metadataEnd + extra) % 32)) % 32;
        String(writer, Key);
        writer.Write(8u);
        String(writer, new string('x', needed));
    }
}
