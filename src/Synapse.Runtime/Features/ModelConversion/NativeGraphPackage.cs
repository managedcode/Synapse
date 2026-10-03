using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

/// <summary>Source-independent scalar execution of prepared native graph packages.</summary>
public static class NativeGraphPackage
{
    /// <summary>Checks the complete package before binding and evaluating a graph.</summary>
    public static IReadOnlyDictionary<string, float[]> Execute(string path, IReadOnlyDictionary<string, float[]> inputs,
        IReadOnlyDictionary<string, long[]> shapes, CancellationToken cancellationToken = default)
    {
        var (model, _) = Read(path, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        return ConversionGraphPipeline.Execute(model, inputs, shapes, cancellationToken);
    }

    /// <summary>Verifies graph, tensor bounds, provenance and full payload integrity.</summary>
    public static ConversionPackageInfo Inspect(string path, CancellationToken cancellationToken = default) =>
        Read(path, cancellationToken).Info;

    internal static uint Version(string path)
    {
        using var stream = File.OpenRead(path);
        Span<byte> header = stackalloc byte[12];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual("SYNAPSE\0"u8))
        {
            throw new InvalidDataException("Model is not a prepared .synapse package.");
        }

        return BinaryPrimitives.ReadUInt32LittleEndian(header[8..]);
    }

    private static (ConversionModel Model, ConversionPackageInfo Info) Read(string path, CancellationToken cancellationToken)
    {
        try
        {
            return ReadCore(path, cancellationToken);
        }
        catch (Exception exception) when (exception is EndOfStreamException or OverflowException or JsonException or ArgumentException)
        {
            throw new InvalidDataException("Native graph package is malformed or truncated.", exception);
        }
    }

    private static (ConversionModel Model, ConversionPackageInfo Info) ReadCore(string path, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var stream = File.OpenRead(path);
        var envelope = new byte[CompiledPackageFormat.EnvelopeLength];
        stream.ReadExactly(envelope);
        var (manifestLength, payloadOffset) = Envelope(envelope, stream.Length);
        var bytes = new byte[manifestLength];
        stream.ReadExactly(bytes);
        if (!CryptographicOperations.FixedTimeEquals(SHA256.HashData(bytes), envelope.AsSpan(32, 32)))
        {
            throw new InvalidDataException("Native graph manifest integrity mismatch.");
        }

        var manifest = JsonSerializer.Deserialize(bytes, NativeGraphJsonContext.Default.NativeGraphManifest)
            ?? throw new InvalidDataException("Native graph manifest is null.");
        Validate(manifest, stream.Length - payloadOffset);
        if (Fingerprint(manifest.Graph) != manifest.GraphFingerprint)
        {
            throw new InvalidDataException("Native graph fingerprint differs from its manifest graph.");
        }

        CheckPadding(stream, payloadOffset);
        if (!CryptographicOperations.FixedTimeEquals(CompiledPackageFormat.HashRange(stream, payloadOffset,
            manifest.PayloadLength, cancellationToken), envelope.AsSpan(64, 32)))
        {
            throw new InvalidDataException("Native graph payload integrity mismatch.");
        }

        var tensors = ReadTensors(stream, manifest.Tensors, payloadOffset, cancellationToken);
        CheckPadding(stream, stream.Length);
        var model = ConversionGraphPipeline.Prepare(new(manifest.SourceFormat, manifest.Graph, tensors), cancellationToken);
        if (Fingerprint(model.Graph) != manifest.GraphFingerprint ||
            !model.Tensors.Select(tensor => tensor.Name).Order(StringComparer.Ordinal).SequenceEqual(manifest.Tensors.Select(tensor => tensor.Name)))
        {
            throw new InvalidDataException("Native graph is not canonical or its fingerprint differs.");
        }

        var info = new ConversionPackageInfo(CompiledPackageFormat.Identity(envelope), manifest.SourceFormat, "native-graph-fp32",
            tensors.Length, stream.Length, manifest.AppliedPasses, model.Graph.Inputs, model.Graph.Outputs, manifest.Sources);
        return (model, info);
    }

    private static (int ManifestLength, long PayloadOffset) Envelope(byte[] envelope, long length)
    {
        var count = BinaryPrimitives.ReadInt32LittleEndian(envelope.AsSpan(12));
        var offset = BinaryPrimitives.ReadInt64LittleEndian(envelope.AsSpan(16));
        if (!envelope.AsSpan(0, 8).SequenceEqual("SYNAPSE\0"u8) || BinaryPrimitives.ReadUInt32LittleEndian(envelope.AsSpan(8)) != 2 ||
            count is <= 0 or > CompiledPackageFormat.MaximumManifestLength ||
            offset != CompiledPackageFormat.Align(CompiledPackageFormat.EnvelopeLength + count) ||
            offset > length || BinaryPrimitives.ReadInt64LittleEndian(envelope.AsSpan(24)) != length)
        {
            throw new InvalidDataException("Native graph package envelope/version is invalid.");
        }

        return (count, offset);
    }

    private static void Validate(NativeGraphManifest manifest, long payloadLength)
    {
        if (manifest.SchemaVersion != 2 || manifest.SourceFormat is not ("onnx" or "safetensors") ||
            manifest.Graph is null || manifest.PayloadLength != payloadLength ||
            manifest.Tensors is null || manifest.Tensors.Length > 1024 || manifest.Sources is null || manifest.Sources.Length is < 1 or > 2 ||
            !CompiledPackageFormat.IsDigest(manifest.GraphFingerprint) || manifest.AppliedPasses is null ||
            !manifest.AppliedPasses.SequenceEqual(["identity-elimination", "dead-code-elimination"]))
        {
            throw new InvalidDataException("Native graph manifest declares invalid bounds or semantics.");
        }

        foreach (var source in manifest.Sources)
        {
            if (source is null || string.IsNullOrWhiteSpace(source.File) || source.File.Contains('\\', StringComparison.Ordinal) ||
                Path.GetFileName(source.File) != source.File || source.File is "." or ".." || source.Length <= 0 ||
                !CompiledPackageFormat.IsDigest(source.Sha256))
            {
                throw new InvalidDataException("Native graph source provenance is invalid.");
            }
        }

        ValidateTensors(manifest.Tensors, payloadLength);
    }

    private static void ValidateTensors(NativeGraphTensor[] tensors, long payloadLength)
    {
        long offset = 0, elements = 0;
        string? previous = null;
        foreach (var tensor in tensors)
        {
            if (tensor is null || string.IsNullOrWhiteSpace(tensor.Name) || tensor.Shape is null || tensor.Shape.Length is < 1 or > 2 ||
                (previous is not null && string.CompareOrdinal(previous, tensor.Name) >= 0))
            {
                throw new InvalidDataException("Native graph tensor declarations are invalid.");
            }

            long count = 1;
            foreach (var dimension in tensor.Shape)
            {
                if (dimension <= 0 || (count = checked(count * dimension)) > 1_000_000)
                {
                    throw new InvalidDataException("Native graph tensor exceeds its element limit.");
                }
            }

            elements = checked(elements + count);
            if (elements > 16_000_000 || tensor.Offset != offset || tensor.ByteLength != checked(count * 4))
            {
                throw new InvalidDataException("Native graph tensor ranges are not canonical.");
            }

            offset = CompiledPackageFormat.Align(checked(offset + tensor.ByteLength));
            previous = tensor.Name;
        }

        if (offset != payloadLength)
        {
            throw new InvalidDataException("Native graph payload length differs from tensor ranges.");
        }
    }

    private static ConversionTensor[] ReadTensors(Stream stream, NativeGraphTensor[] descriptors, long payloadOffset, CancellationToken cancellationToken)
    {
        var tensors = new ConversionTensor[descriptors.Length];
        stream.Position = payloadOffset;
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        for (var index = 0; index < descriptors.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var descriptor = descriptors[index];
            CheckPadding(stream, checked(payloadOffset + descriptor.Offset));
            var data = new float[checked((int)(descriptor.ByteLength / 4))];
            for (var element = 0; element < data.Length; element++)
            {
                if (element % 16384 == 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }

                data[element] = reader.ReadSingle();
                if (!float.IsFinite(data[element]))
                {
                    throw new InvalidDataException("Native graph weight contains a nonfinite value.");
                }
            }

            tensors[index] = new(descriptor.Name, descriptor.Shape, data);
        }

        return tensors;
    }

    internal static string Fingerprint(ConversionGraph graph) => Convert.ToHexStringLower(
        SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(graph, ConversionJsonContext.Default.ConversionGraph)));

    private static void CheckPadding(Stream stream, long target)
    {
        var count = target - stream.Position;
        if (count is < 0 or >= CompiledPackageFormat.Alignment)
        {
            throw new InvalidDataException("Native graph alignment is invalid.");
        }

        for (var index = 0L; index < count; index++)
        {
            if (stream.ReadByte() != 0)
            {
                throw new InvalidDataException("Native graph alignment bytes must be zero.");
            }
        }
    }
}
