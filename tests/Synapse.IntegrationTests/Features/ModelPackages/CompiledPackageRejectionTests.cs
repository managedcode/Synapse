using System.Buffers.Binary;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

[NotInParallel]
public sealed class CompiledPackageRejectionTests
{
    [Test]
    public async Task RuntimeRequiresExplicitCompiledArtifact()
    {
        using var fixture = new CompiledPackageFixture();
        await Assert.That(() => ModelLoader.Load(fixture.Source, 16)).Throws<NotSupportedException>();
        await Assert.That(() => Qwen2Model.Load(fixture.Source, 16)).Throws<NotSupportedException>();
        await Assert.That(() => TextTokenizers.FromModel(fixture.Source)).Throws<NotSupportedException>();
        await Assert.That(File.Exists(fixture.Destination)).IsFalse();
    }

    [Test]
    [Arguments("version")]
    [Arguments("truncated")]
    [Arguments("manifest")]
    [Arguments("payload")]
    public async Task ReaderRejectsEnvelopeAndPayloadCorruption(string kind)
    {
        using var fixture = new CompiledPackageFixture();
        await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        var bytes = File.ReadAllBytes(fixture.Destination);
        switch (kind)
        {
            case "version": BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8), 2); break;
            case "truncated": bytes = bytes[..^1]; break;
            case "manifest": bytes[100] ^= 1; break;
            case "payload": bytes[^1] ^= 1; break;
            default: throw new ArgumentOutOfRangeException(nameof(kind));
        }

        File.WriteAllBytes(fixture.Destination, bytes);
        await Assert.That(() => CompiledPackageReader.Inspect(fixture.Destination)).Throws<InvalidDataException>();
    }

    [Test]
    [Arguments("offset")]
    [Arguments("overlap")]
    [Arguments("source-offset")]
    [Arguments("encoding")]
    [Arguments("dimensions")]
    [Arguments("duplicate")]
    [Arguments("graph")]
    [Arguments("null-dimensions")]
    [Arguments("oversized-header")]
    public async Task ReaderRejectsMalformedIndexEvenWhenDigestsAreRecomputed(string kind)
    {
        using var fixture = new CompiledPackageFixture();
        await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        RewriteManifest(fixture.Destination, manifest => Mutate(manifest, kind));
        await Assert.That(() => CompiledPackageReader.Inspect(fixture.Destination)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task FailedPublicationAndCancellationPreserveExistingDestination()
    {
        using var fixture = new CompiledPackageFixture();
        var sentinel = new byte[] { 1, 2, 3, 4 };
        File.WriteAllBytes(fixture.Destination, sentinel);
        await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination)).Throws<IOException>();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination, cancellation.Token))
            .Throws<OperationCanceledException>();
        await Assert.That(File.ReadAllBytes(fixture.Destination).SequenceEqual(sentinel))
            .IsTrue().Because("Failed publication and cancellation must leave the destination byte-for-byte unchanged.");
        await Assert.That(Directory.EnumerateFiles(Path.GetDirectoryName(fixture.Destination)!,
            Path.GetFileName(fixture.Destination) + ".*.partial.synapse").Any()).IsFalse();
    }

    [Test]
    public async Task CompilerRejectsIdenticalPathsWithoutChangingSource()
    {
        using var fixture = new CompiledPackageFixture();
        var before = File.ReadAllBytes(fixture.Source);
        await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Source)).Throws<ArgumentException>();
        await Assert.That(File.ReadAllBytes(fixture.Source).SequenceEqual(before))
            .IsTrue().Because("Rejecting identical source and output paths must preserve the complete source byte sequence.");
    }

    [Test]
    public async Task AtomicMoveFailureDeletesCompiledTemporaryFile()
    {
        using var fixture = new CompiledPackageFixture();
        _ = Directory.CreateDirectory(fixture.Destination);
        try
        {
            await Assert.That(async () => await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination)).Throws<IOException>();
            await Assert.That(Directory.Exists(fixture.Destination)).IsTrue();
            await Assert.That(Directory.EnumerateFiles(Path.GetDirectoryName(fixture.Destination)!,
                Path.GetFileName(fixture.Destination) + ".*.partial.synapse").Any()).IsFalse();
        }
        finally
        {
            Directory.Delete(fixture.Destination);
        }
    }

    private static CompiledPackageManifest Mutate(CompiledPackageManifest manifest, string kind)
    {
        var tensors = manifest.Tensors.ToArray();
        tensors[0] = kind switch
        {
            "offset" => tensors[0] with { Offset = tensors[0].Offset + 1 },
            "source-offset" => tensors[0] with { SourceOffset = tensors[0].SourceOffset + 32 },
            "encoding" => tensors[0] with { Encoding = 999 },
            "dimensions" => tensors[0] with { Dimensions = [1, 1] },
            "null-dimensions" => tensors[0] with { Dimensions = null! },
            _ => tensors[0],
        };
        if (kind == "overlap")
        {
            tensors[1] = tensors[1] with { Offset = tensors[0].Offset };
        }

        if (kind == "duplicate")
        {
            tensors[1] = tensors[0];
        }

        return manifest with
        {
            Tensors = tensors,
            GraphFingerprint = kind == "graph" ? new string('0', 64) : manifest.GraphFingerprint,
            HeaderLength = kind == "oversized-header" ? long.MaxValue : manifest.HeaderLength,
        };
    }

    private static void RewriteManifest(string path, Func<CompiledPackageManifest, CompiledPackageManifest> change)
    {
        var bytes = File.ReadAllBytes(path);
        var length = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(12));
        var oldOffset = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(16));
        var manifest = JsonSerializer.Deserialize(bytes.AsSpan(96, length), CompiledPackageJsonContext.Default.CompiledPackageManifest)!;
        var replacement = JsonSerializer.SerializeToUtf8Bytes(change(manifest), CompiledPackageJsonContext.Default.CompiledPackageManifest);
        var newOffset = CompiledPackageFormat.Align(96 + replacement.Length);
        var output = new byte[checked((int)(newOffset + bytes.Length - oldOffset))];
        replacement.CopyTo(output, 96);
        bytes.AsSpan((int)oldOffset).CopyTo(output.AsSpan((int)newOffset));
        CompiledPackageFormat.Envelope(replacement, newOffset, output.Length, bytes.AsSpan(64, 32).ToArray()).CopyTo(output, 0);
        File.WriteAllBytes(path, output);
    }
}
