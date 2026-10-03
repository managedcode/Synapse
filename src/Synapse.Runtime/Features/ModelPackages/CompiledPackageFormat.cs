using System.Buffers.Binary;
using System.Security.Cryptography;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

internal static class CompiledPackageFormat
{
    internal const int EnvelopeLength = 96;
    internal const int Alignment = 64;
    internal const int MaximumManifestLength = 4 * 1024 * 1024;
    internal const int MaximumHeaderLength = 64 * 1024 * 1024;
    internal const int MaximumTensorCount = 16_384;

    internal static long Align(long length) => checked((length + Alignment - 1) / Alignment * Alignment);

    internal static byte[] Envelope(byte[] manifest, long payloadOffset, long fileLength, byte[] payloadHash)
    {
        var bytes = new byte[EnvelopeLength];
        "SYNAPSE\0"u8.CopyTo(bytes);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8), 1);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12), manifest.Length);
        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(16), payloadOffset);
        BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(24), fileLength);
        SHA256.HashData(manifest).CopyTo(bytes, 32);
        payloadHash.CopyTo(bytes, 64);
        return bytes;
    }

    internal static byte[] HashRange(Stream stream, long offset, long count, CancellationToken cancellationToken)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1 << 16];
        stream.Position = offset;
        while (count > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var read = stream.Read(buffer.AsSpan(0, (int)Math.Min(count, buffer.Length)));
            if (read == 0)
            {
                throw new InvalidDataException("Compiled package payload is truncated.");
            }

            hash.AppendData(buffer, 0, read);
            count -= read;
        }

        cancellationToken.ThrowIfCancellationRequested();
        return hash.GetHashAndReset();
    }

    internal static bool IsDigest(string? value) => value?.Length == 64 && value.All(Uri.IsHexDigit);

    internal static string Identity(ReadOnlySpan<byte> envelope) =>
        Convert.ToHexStringLower(SHA256.HashData(envelope));
}
