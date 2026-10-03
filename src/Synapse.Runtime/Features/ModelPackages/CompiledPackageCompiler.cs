using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

/// <summary>Losslessly packs verified Qwen2 GGUF tensor bytes into a directly executable .synapse package.</summary>
public static class CompiledPackageCompiler
{
    /// <summary>Streams compilation and atomically publishes a new destination; existing files are never overwritten.</summary>
    public static async Task<CompiledPackageInfo> CompileAsync(string sourcePath, string destinationPath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);
        var sourceFullPath = Path.GetFullPath(sourcePath);
        var destinationFullPath = Path.GetFullPath(destinationPath);
        if (string.Equals(sourceFullPath, destinationFullPath, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("Source and compiled destination paths must differ.", nameof(destinationPath));
        }

        if (!string.Equals(Path.GetExtension(sourceFullPath), ".gguf", StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(Path.GetExtension(destinationFullPath), ".synapse", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException("Compilation requires a .gguf source and a .synapse destination.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        if (File.Exists(destinationFullPath))
        {
            throw new IOException("Compiled destination already exists.");
        }

        var directory = Path.GetDirectoryName(destinationFullPath)!;
        _ = Directory.CreateDirectory(directory);
        var temporary = destinationFullPath + $".{Guid.NewGuid():N}.partial.synapse";
        try
        {
            await WriteAsync(sourceFullPath, temporary, cancellationToken).ConfigureAwait(false);
            var info = CompiledPackageReader.Inspect(temporary, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, destinationFullPath, overwrite: false);
            return info;
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    private static async Task WriteAsync(string sourcePath, string temporary, CancellationToken cancellationToken)
    {
        await using var source = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.Asynchronous);
        using var boundedHeader = new CompiledReadStream(source, 0, Math.Min(source.Length, CompiledPackageFormat.MaximumHeaderLength));
        var descriptor = GgufHeaderReader.Read(boundedHeader, source.Length, CompiledPackageFormat.MaximumTensorCount, CompiledPackageFormat.MaximumTensorCount);
        CompiledPackageValidation.Source(descriptor);
        using var file = GgufFile.OpenMapped(source, descriptor, Path.GetFileName(sourcePath));
        var graph = CompiledPackageValidation.Graph(file);
        source.Position = 0;
        var sourceHash = await SHA256.HashDataAsync(source, cancellationToken).ConfigureAwait(false);
        var tensors = PackIndex(CompiledPackageValidation.OrderedTensors(descriptor, graph), descriptor.DataOffset, out var payloadLength);
        var manifest = new CompiledPackageManifest(1, "qwen2", file.SourceFile, source.Length, Convert.ToHexStringLower(sourceHash),
            ModelGraphFingerprint.Compute(graph).Value, descriptor.DataOffset, payloadLength, tensors);
        var manifestBytes = JsonSerializer.SerializeToUtf8Bytes(manifest, CompiledPackageJsonContext.Default.CompiledPackageManifest);
        if (manifestBytes.Length > CompiledPackageFormat.MaximumManifestLength)
        {
            throw new InvalidDataException("Compiled manifest exceeds its size limit.");
        }

        var payloadOffset = CompiledPackageFormat.Align(CompiledPackageFormat.EnvelopeLength + manifestBytes.Length);
        var buffer = new byte[1 << 16];
        await using var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 1 << 16, FileOptions.Asynchronous);
        await output.WriteAsync(new byte[CompiledPackageFormat.EnvelopeLength], cancellationToken).ConfigureAwait(false);
        await output.WriteAsync(manifestBytes, cancellationToken).ConfigureAwait(false);
        await PadAsync(output, payloadOffset, cancellationToken).ConfigureAwait(false);
        await CopyRangeAsync(source, output, buffer, 0, descriptor.DataOffset, cancellationToken).ConfigureAwait(false);
        foreach (var tensor in tensors)
        {
            await PadAsync(output, checked(payloadOffset + tensor.Offset), cancellationToken).ConfigureAwait(false);
            await CopyRangeAsync(source, output, buffer, tensor.SourceOffset, tensor.ByteLength, cancellationToken).ConfigureAwait(false);
        }

        await PadAsync(output, checked(payloadOffset + payloadLength), cancellationToken).ConfigureAwait(false);
        await FinishAsync(source, output, manifestBytes, sourceHash, payloadOffset, cancellationToken).ConfigureAwait(false);
    }

    private static CompiledTensor[] PackIndex(GgufTensorInfo[] tensors, long headerLength, out long payloadLength)
    {
        var offset = CompiledPackageFormat.Align(headerLength);
        var index = new CompiledTensor[tensors.Length];
        for (var tensorIndex = 0; tensorIndex < tensors.Length; tensorIndex++)
        {
            var tensor = tensors[tensorIndex];
            index[tensorIndex] = new(tensor.Name, tensor.Offset, offset, tensor.ByteLength, tensor.Type, tensor.Dimensions);
            offset = CompiledPackageFormat.Align(checked(offset + tensor.ByteLength));
        }

        payloadLength = offset;
        return index;
    }

    private static async Task CopyRangeAsync(Stream source, Stream output, byte[] buffer, long offset, long count, CancellationToken cancellationToken)
    {
        source.Position = offset;
        while (count > 0)
        {
            var length = (int)Math.Min(count, buffer.Length);
            await source.ReadExactlyAsync(buffer.AsMemory(0, length), cancellationToken).ConfigureAwait(false);
            await output.WriteAsync(buffer.AsMemory(0, length), cancellationToken).ConfigureAwait(false);
            count -= length;
        }
    }

    private static async Task PadAsync(Stream output, long target, CancellationToken cancellationToken)
    {
        var count = checked((int)(target - output.Position));
        if (count is < 0 or >= CompiledPackageFormat.Alignment)
        {
            throw new InvalidDataException("Compiler alignment boundary is inconsistent.");
        }

        await output.WriteAsync(new byte[count], cancellationToken).ConfigureAwait(false);
    }

    private static async Task FinishAsync(Stream source, FileStream output, byte[] manifest, byte[] sourceHash, long payloadOffset, CancellationToken cancellationToken)
    {
        source.Position = 0;
        var finalSourceHash = await SHA256.HashDataAsync(source, cancellationToken).ConfigureAwait(false);
        if (!CryptographicOperations.FixedTimeEquals(sourceHash, finalSourceHash))
        {
            throw new InvalidDataException("Source changed while compiling.");
        }

        await output.FlushAsync(cancellationToken).ConfigureAwait(false);
        var payloadHash = CompiledPackageFormat.HashRange(output, payloadOffset, output.Length - payloadOffset, cancellationToken);
        var envelope = CompiledPackageFormat.Envelope(manifest, payloadOffset, output.Length, payloadHash);
        output.Position = 0;
        await output.WriteAsync(envelope, cancellationToken).ConfigureAwait(false);
        await output.FlushAsync(cancellationToken).ConfigureAwait(false);
        output.Flush(flushToDisk: true);
    }
}
