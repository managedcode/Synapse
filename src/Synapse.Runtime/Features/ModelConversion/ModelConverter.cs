using System.Security.Cryptography;
using System.Buffers.Binary;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

/// <summary>Explicit source preparation options; no runtime conversion is implicit.</summary>
public sealed record ModelConversionOptions(string? GraphPath = null, IReadOnlyDictionary<string, ConversionBound>? Dimensions = null);

/// <summary>Format-neutral explicit model preparation.</summary>
public static class ModelConverter
{
    /// <summary>Converts a supported source to a new prepared artifact.</summary>
    public static Task<ConversionPackageInfo> ConvertAsync(string sourcePath, string destinationPath,
        ModelConversionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);
        var source = Path.GetFullPath(sourcePath);
        var destination = Path.GetFullPath(destinationPath);
        if (!string.Equals(Path.GetExtension(destination), ".synapse", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(source, destination, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("Conversion requires a distinct .synapse destination.", nameof(destinationPath));
        }

        cancellationToken.ThrowIfCancellationRequested();
        if (File.Exists(destination) || Directory.Exists(destination))
        {
            throw new IOException("Conversion destination already exists.");
        }

        options ??= new();
        var format = Path.GetExtension(source).ToLowerInvariant();
        return format switch
        {
            ".gguf" when options.GraphPath is null && options.Dimensions is null => ConvertGgufAsync(source, destination, cancellationToken),
            ".onnx" when options.GraphPath is null => ConvertGraphAsync(source, destination, options, cancellationToken),
            ".safetensors" when options.GraphPath is not null && options.Dimensions is null => ConvertGraphAsync(source, destination, options, cancellationToken),
            _ => throw new NotSupportedException("Supported conversion sources: .gguf, .onnx [--dimension symbol=min:max], " +
                ".safetensors --graph graph.json. Unsupported source/option combination."),
        };
    }

    /// <summary>Verifies either version of the prepared package.</summary>
    public static ConversionPackageInfo Inspect(string path, CancellationToken cancellationToken = default) => NativeGraphPackage.Version(path) switch
    {
        1 => FromGguf(CompiledPackageReader.Inspect(path, cancellationToken), path),
        2 => NativeGraphPackage.Inspect(path, cancellationToken),
        _ => throw new InvalidDataException("Unknown .synapse package version."),
    };

    private static async Task<ConversionPackageInfo> ConvertGgufAsync(string source, string destination, CancellationToken cancellationToken) =>
        FromGguf(await CompiledPackageCompiler.CompileAsync(source, destination, cancellationToken).ConfigureAwait(false), destination);

    private static ConversionPackageInfo FromGguf(CompiledPackageInfo info, string path)
    {
        using var stream = File.OpenRead(path);
        Span<byte> envelope = stackalloc byte[CompiledPackageFormat.EnvelopeLength];
        stream.ReadExactly(envelope);
        var length = BinaryPrimitives.ReadInt32LittleEndian(envelope[12..]);
        if (length is <= 0 or > CompiledPackageFormat.MaximumManifestLength)
        {
            throw new InvalidDataException("Prepared GGUF manifest length is invalid.");
        }

        var bytes = new byte[length];
        stream.ReadExactly(bytes);
        var manifest = JsonSerializer.Deserialize(bytes, CompiledPackageJsonContext.Default.CompiledPackageManifest)
            ?? throw new InvalidDataException("Prepared GGUF manifest is null.");
        if (manifest.SourceSha256 != info.SourceSha256 || CompiledPackageFormat.Identity(envelope) != info.Identity)
        {
            throw new InvalidDataException("Prepared GGUF changed while inspecting provenance.");
        }

        return new(info.Identity, "gguf", info.Architecture, info.TensorCount, info.Length,
            ["lossless-first-use-tensor-packing"], [], [], [new(info.SourceFile, manifest.SourceLength, info.SourceSha256)]);
    }

    private static async Task<ConversionPackageInfo> ConvertGraphAsync(string source, string destination,
        ModelConversionOptions options, CancellationToken cancellationToken)
    {
        string[] paths = options.GraphPath is null ? [source] : [source, Path.GetFullPath(options.GraphPath)];
        if (paths.Any(path => string.Equals(path, destination, StringComparison.OrdinalIgnoreCase)))
        {
            throw new ArgumentException("Graph source and destination paths must differ.", nameof(destination));
        }

        if (new FileInfo(source).Length > (options.GraphPath is null ? 64L : 128L) * 1024 * 1024 ||
            (paths.Length == 2 && new FileInfo(paths[1]).Length > 4L * 1024 * 1024))
        {
            throw new InvalidDataException("Source exceeds the bounded graph conversion size limit.");
        }

        var identities = await IdentitiesAsync(paths, cancellationToken).ConfigureAwait(false);
        var model = options.GraphPath is null ? OnnxModelImporter.Import(source, options.Dimensions, cancellationToken) :
            SafeTensorsModelImporter.Import(source, paths[1], cancellationToken);
        var prepared = ConversionGraphPipeline.Prepare(model, cancellationToken);
        _ = Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var temporary = destination + $".{Guid.NewGuid():N}.partial.synapse";
        try
        {
            await NativeGraphWriter.WriteAsync(temporary, prepared, identities, cancellationToken).ConfigureAwait(false);
            var info = NativeGraphPackage.Inspect(temporary, cancellationToken);
            var after = await IdentitiesAsync(paths, cancellationToken).ConfigureAwait(false);
            if (!identities.SequenceEqual(after))
            {
                throw new InvalidDataException("Source changed during model conversion.");
            }

            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, destination, overwrite: false);
            return info;
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    private static async Task<ConversionSource[]> IdentitiesAsync(string[] paths, CancellationToken cancellationToken)
    {
        var sources = new ConversionSource[paths.Length];
        for (var index = 0; index < paths.Length; index++)
        {
            await using var stream = new FileStream(paths[index], FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.Asynchronous);
            var length = stream.Length;
            var digest = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
            if (stream.Length != length)
            {
                throw new InvalidDataException("Source length changed while hashing.");
            }

            sources[index] = new(Path.GetFileName(paths[index]), length, Convert.ToHexStringLower(digest));
        }

        return sources;
    }
}

/// <summary>Source content identity retained independently of source availability.</summary>
public sealed record ConversionSource(string File, long Length, string Sha256);

/// <summary>Verified package identity, native graph boundaries and transformation report.</summary>
public sealed record ConversionPackageInfo(string Identity, string SourceFormat, string Architecture,
    int TensorCount, long Length, string[] AppliedPasses, ConversionInput[] Inputs, string[] Outputs, ConversionSource[] Sources);
