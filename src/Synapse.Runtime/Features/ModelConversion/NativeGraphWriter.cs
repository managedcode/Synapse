using System.Buffers.Binary;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal static class NativeGraphWriter
{
    internal static async Task WriteAsync(string path, ConversionModel model, ConversionSource[] sources, CancellationToken cancellationToken)
    {
        var tensors = model.Tensors.OrderBy(tensor => tensor.Name, StringComparer.Ordinal).ToArray();
        var descriptors = new NativeGraphTensor[tensors.Length];
        var payloadLength = 0L;
        for (var index = 0; index < tensors.Length; index++)
        {
            var tensor = tensors[index];
            var length = checked((long)tensor.Data.Length * 4);
            descriptors[index] = new(tensor.Name, tensor.Shape, payloadLength, length);
            payloadLength = CompiledPackageFormat.Align(checked(payloadLength + length));
        }

        var manifest = new NativeGraphManifest(2, model.SourceFormat, model.Graph, NativeGraphPackage.Fingerprint(model.Graph), sources,
            ["identity-elimination", "dead-code-elimination"], descriptors, payloadLength);
        var bytes = JsonSerializer.SerializeToUtf8Bytes(manifest, NativeGraphJsonContext.Default.NativeGraphManifest);
        if (bytes.Length > CompiledPackageFormat.MaximumManifestLength)
        {
            throw new InvalidDataException("Native graph manifest exceeds its size limit.");
        }

        var offset = CompiledPackageFormat.Align(CompiledPackageFormat.EnvelopeLength + bytes.Length);
        await using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 1 << 16, FileOptions.Asynchronous);
        await stream.WriteAsync(new byte[CompiledPackageFormat.EnvelopeLength], cancellationToken).ConfigureAwait(false);
        await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
        await PadAsync(stream, offset, cancellationToken).ConfigureAwait(false);
        var buffer = new byte[1 << 16];
        for (var index = 0; index < tensors.Length; index++)
        {
            await PadAsync(stream, checked(offset + descriptors[index].Offset), cancellationToken).ConfigureAwait(false);
            await WriteTensorAsync(stream, tensors[index].Data, buffer, cancellationToken).ConfigureAwait(false);
        }

        await PadAsync(stream, checked(offset + payloadLength), cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        var hash = CompiledPackageFormat.HashRange(stream, offset, payloadLength, cancellationToken);
        var envelope = CompiledPackageFormat.Envelope(bytes, offset, stream.Length, hash);
        BinaryPrimitives.WriteUInt32LittleEndian(envelope.AsSpan(8), 2);
        stream.Position = 0;
        await stream.WriteAsync(envelope, cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        stream.Flush(flushToDisk: true);
    }

    private static async Task WriteTensorAsync(Stream stream, float[] values, byte[] buffer, CancellationToken cancellationToken)
    {
        var position = 0;
        while (position < values.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var count = Math.Min(values.Length - position, buffer.Length / 4);
            Encode(values, position, count, buffer);
            await stream.WriteAsync(buffer.AsMemory(0, count * 4), cancellationToken).ConfigureAwait(false);
            position += count;
        }
    }

    private static void Encode(float[] values, int position, int count, byte[] buffer)
    {
        for (var index = 0; index < count; index++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(index * 4, 4), values[position + index]);
        }
    }

    private static async Task PadAsync(Stream stream, long target, CancellationToken cancellationToken)
    {
        var count = target - stream.Position;
        if (count is < 0 or >= CompiledPackageFormat.Alignment)
        {
            throw new InvalidDataException("Native graph writer alignment is invalid.");
        }

        await stream.WriteAsync(new byte[(int)count], cancellationToken).ConfigureAwait(false);
    }
}
