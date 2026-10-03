using System.Buffers.Binary;
using System.Text;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

[NotInParallel]
public sealed class CompiledPackageSourceTests
{
    [Test]
    [Arguments("misaligned")]
    [Arguments("overlap")]
    [Arguments("truncated")]
    public async Task CompilerRejectsInvalidSourceRangesWithoutPublishing(string kind)
    {
        using var fixture = new CompiledPackageFixture();
        var bytes = File.ReadAllBytes(fixture.Source);
        var offsets = TensorOffsetFields(bytes);
        switch (kind)
        {
            case "misaligned": BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(offsets[0]), 1); break;
            case "overlap": BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(offsets[1]), 0); break;
            case "truncated": bytes = bytes[..(bytes.Length / 2)]; break;
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }

        File.WriteAllBytes(fixture.Source, bytes);
        await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination)).Throws<InvalidDataException>();
        await Assert.That(File.Exists(fixture.Destination)).IsFalse();
        await Assert.That(Directory.EnumerateFiles(Path.GetDirectoryName(fixture.Destination)!,
            Path.GetFileName(fixture.Destination) + ".*.partial.synapse").Any()).IsFalse();
    }

    [Test]
    [Arguments("encoding")]
    [Arguments("architecture")]
    public async Task CompilerRejectsUnsupportedSourcesExplicitly(string kind)
    {
        using var fixture = new CompiledPackageFixture();
        var bytes = File.ReadAllBytes(fixture.Source);
        if (kind == "encoding")
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(TensorOffsetFields(bytes)[0] - 4), 999);
        }
        else
        {
            var offset = bytes.AsSpan().IndexOf("qwen2"u8);
            "other"u8.CopyTo(bytes.AsSpan(offset));
        }

        File.WriteAllBytes(fixture.Source, bytes);
        await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination)).Throws<NotSupportedException>();
        await Assert.That(File.Exists(fixture.Destination)).IsFalse();
    }

    private static int[] TensorOffsetFields(byte[] bytes)
    {
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        stream.Position = 8;
        var tensors = checked((int)reader.ReadUInt64());
        var metadata = checked((int)reader.ReadUInt64());
        for (var index = 0; index < metadata; index++)
        {
            _ = GgufMetadataValues.ReadString(reader);
            _ = GgufMetadataValues.ReadValue(reader, reader.ReadUInt32(), 0);
        }

        var offsets = new int[tensors];
        for (var index = 0; index < tensors; index++)
        {
            _ = GgufMetadataValues.ReadString(reader);
            var rank = reader.ReadUInt32();
            stream.Position += (rank * 8) + 4;
            offsets[index] = checked((int)stream.Position);
            stream.Position += 8;
        }

        return offsets;
    }
}
