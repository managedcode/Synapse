using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

/// <summary>Validates the bounded, inert lossless compiled format before opening a model.</summary>
public static class CompiledPackageReader
{
    /// <summary>Checks structure, source-index agreement, graph identity, and complete payload integrity.</summary>
    public static CompiledPackageInfo Inspect(string path, CancellationToken cancellationToken = default)
    {
        var package = Open(path, cancellationToken);
        using (package.File)
        {
            return package.Info;
        }
    }

    internal static CompiledPackageDescriptor Open(string path, CancellationToken cancellationToken = default)
    {
        try
        {
            return ReadCore(path, cancellationToken);
        }
        catch (Exception exception) when (exception is EndOfStreamException or OverflowException or JsonException or ArgumentException)
        {
            throw new InvalidDataException("Compiled package contains a malformed or truncated index.", exception);
        }
    }

    private static CompiledPackageDescriptor ReadCore(string path, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        var envelope = new byte[CompiledPackageFormat.EnvelopeLength];
        stream.ReadExactly(envelope);
        var (manifestLength, payloadOffset) = ValidateEnvelope(envelope, stream.Length);
        var bytes = new byte[manifestLength];
        stream.ReadExactly(bytes);
        if (!CryptographicOperations.FixedTimeEquals(SHA256.HashData(bytes), envelope.AsSpan(32, 32)))
        {
            throw new InvalidDataException("Compiled package manifest integrity mismatch.");
        }

        var manifest = JsonSerializer.Deserialize(bytes, CompiledPackageJsonContext.Default.CompiledPackageManifest)
            ?? throw new InvalidDataException("Compiled package manifest is null.");
        ValidateManifest(manifest, stream.Length - payloadOffset);
        var padding = new byte[checked((int)(payloadOffset - stream.Position))];
        stream.ReadExactly(padding);
        if (padding.Any(value => value != 0) || !CryptographicOperations.FixedTimeEquals(
            CompiledPackageFormat.HashRange(stream, payloadOffset, manifest.PayloadLength, cancellationToken), envelope.AsSpan(64, 32)))
        {
            throw new InvalidDataException("Compiled package payload integrity mismatch.");
        }

        using var header = new CompiledReadStream(stream, payloadOffset, manifest.HeaderLength);
        var source = GgufHeaderReader.Read(header, manifest.SourceLength, CompiledPackageFormat.MaximumTensorCount, CompiledPackageFormat.MaximumTensorCount);
        var descriptor = CompiledPackageValidation.Rebase(source, manifest, payloadOffset);
        var file = GgufFile.OpenMapped(stream, descriptor, manifest.SourceFile);
        try
        {
            return Finish(file, source, manifest, envelope, stream.Length);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    private static CompiledPackageDescriptor Finish(GgufFile file, GgufDescriptor source, CompiledPackageManifest manifest, byte[] envelope, long length)
    {
        var graph = CompiledPackageValidation.Graph(file);
        if (ModelGraphFingerprint.Compute(graph).Value != manifest.GraphFingerprint ||
            !CompiledPackageValidation.OrderedTensors(source, graph).Select(tensor => tensor.Name)
                .SequenceEqual(manifest.Tensors.Select(tensor => tensor.Name)))
        {
            throw new InvalidDataException("Compiled package graph fingerprint or tensor order mismatch.");
        }

        var info = new CompiledPackageInfo(CompiledPackageFormat.Identity(envelope), manifest.Architecture,
            manifest.SourceFile, manifest.SourceSha256, manifest.GraphFingerprint, manifest.Tensors.Length, length);
        return new CompiledPackageDescriptor(info, file);
    }

    private static (int ManifestLength, long PayloadOffset) ValidateEnvelope(byte[] envelope, long length)
    {
        var manifestLength = BinaryPrimitives.ReadInt32LittleEndian(envelope.AsSpan(12));
        var payloadOffset = BinaryPrimitives.ReadInt64LittleEndian(envelope.AsSpan(16));
        if (!envelope.AsSpan(0, 8).SequenceEqual("SYNAPSE\0"u8) ||
            BinaryPrimitives.ReadUInt32LittleEndian(envelope.AsSpan(8)) != 1 ||
            manifestLength is <= 0 or > CompiledPackageFormat.MaximumManifestLength ||
            payloadOffset != CompiledPackageFormat.Align(CompiledPackageFormat.EnvelopeLength + manifestLength) ||
            payloadOffset >= length || BinaryPrimitives.ReadInt64LittleEndian(envelope.AsSpan(24)) != length)
        {
            throw new InvalidDataException("Compiled package has an unknown version, invalid envelope, or length.");
        }

        return (manifestLength, payloadOffset);
    }

    private static void ValidateManifest(CompiledPackageManifest manifest, long payloadLength)
    {
        if (manifest.SchemaVersion != 1 || manifest.Architecture != "qwen2" ||
            string.IsNullOrWhiteSpace(manifest.SourceFile) || manifest.SourceFile.Contains('\\', StringComparison.Ordinal) ||
            Path.GetFileName(manifest.SourceFile) != manifest.SourceFile || manifest.SourceFile is "." or ".." ||
            manifest.SourceLength <= 0 || !CompiledPackageFormat.IsDigest(manifest.SourceSha256) ||
            !CompiledPackageFormat.IsDigest(manifest.GraphFingerprint) ||
            manifest.HeaderLength is <= 0 or > CompiledPackageFormat.MaximumHeaderLength ||
            manifest.HeaderLength > manifest.SourceLength || manifest.PayloadLength != payloadLength ||
            manifest.Tensors is null || manifest.Tensors.Length is <= 0 or > CompiledPackageFormat.MaximumTensorCount)
        {
            throw new InvalidDataException("Compiled package manifest exceeds bounds or declares unsupported semantics.");
        }
    }
}

internal sealed record CompiledPackageDescriptor(CompiledPackageInfo Info, GgufFile File);
