using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

internal enum FloatSourceKind
{
    Fp32,
    Fp16,
    Bf16,
}

/// <summary>Whole-tensor float decoding without a delegate or block slice per scalar.</summary>
internal sealed class FloatSourceDecoder(string encodingId, FloatSourceKind kind, int blockBytes) : ISourceTensorDecoder
{
    public string EncodingId { get; } = encodingId;

    public int BlockElements => 1;

    public int BlockBytes { get; } = blockBytes;

    public long GetByteLength(long elements) => elements > 0
        ? checked(elements * BlockBytes)
        : throw new ArgumentException($"{EncodingId} needs a positive multiple of 1 elements.", nameof(elements));

    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (source.Length != GetByteLength(destination.Length))
        {
            throw new ArgumentException($"{EncodingId} needs {GetByteLength(destination.Length)} bytes for {destination.Length} values.");
        }

        if (kind == FloatSourceKind.Fp32)
        {
            ValidateFp32(source);
            WriteFp32(source, destination);
            return;
        }

        Validate16(source, kind == FloatSourceKind.Bf16 ? (ushort)0x7F80 : (ushort)0x7C00);
        if (kind == FloatSourceKind.Bf16)
        {
            WriteBf16(source, destination);
        }
        else
        {
            WriteFp16(source, destination);
        }
    }

    private void ValidateFp32(ReadOnlySpan<byte> source)
    {
        if (BitConverter.IsLittleEndian)
        {
            var index = FloatSourceValidation.FirstNonFinite32(MemoryMarshal.Cast<byte, uint>(source));
            if (index >= 0)
            {
                RejectNonFinite(index);
            }

            return;
        }

        for (var index = 0; index < source.Length / sizeof(uint); index++)
        {
            if ((BinaryPrimitives.ReadUInt32LittleEndian(source[(index * sizeof(uint))..]) & 0x7F80_0000) == 0x7F80_0000)
            {
                RejectNonFinite(index);
            }
        }
    }

    private void Validate16(ReadOnlySpan<byte> source, ushort exponentMask)
    {
        if (BitConverter.IsLittleEndian)
        {
            var index = FloatSourceValidation.FirstNonFinite16(MemoryMarshal.Cast<byte, ushort>(source), exponentMask);
            if (index >= 0)
            {
                RejectNonFinite(index);
            }

            return;
        }

        for (var index = 0; index < source.Length / sizeof(ushort); index++)
        {
            if ((BinaryPrimitives.ReadUInt16LittleEndian(source[(index * sizeof(ushort))..]) & exponentMask) == exponentMask)
            {
                RejectNonFinite(index);
            }
        }
    }

    private static void WriteFp32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var outputBytes = MemoryMarshal.AsBytes(destination);
        if (BitConverter.IsLittleEndian && !source.Overlaps(outputBytes))
        {
            source.CopyTo(outputBytes);
            return;
        }

        // Preserve the block decoder's forward writes when caller storage overlaps.
        for (var index = 0; index < destination.Length; index++)
        {
            destination[index] = BinaryPrimitives.ReadSingleLittleEndian(source[(index * sizeof(float))..]);
        }
    }

    private static void WriteBf16(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var index = 0;
        if (BitConverter.IsLittleEndian)
        {
            var codes = MemoryMarshal.Cast<byte, ushort>(source);
            if (Vector.IsHardwareAccelerated && !source.Overlaps(MemoryMarshal.AsBytes(destination)))
            {
                for (; index <= codes.Length - Vector<ushort>.Count; index += Vector<ushort>.Count)
                {
                    Vector.Widen(new Vector<ushort>(codes[index..]), out var low, out var high);
                    Vector.AsVectorSingle(Vector.ShiftLeft(low, 16)).CopyTo(destination[index..]);
                    Vector.AsVectorSingle(Vector.ShiftLeft(high, 16)).CopyTo(destination[(index + Vector<uint>.Count)..]);
                }
            }

            for (; index < codes.Length; index++)
            {
                destination[index] = BitConverter.UInt32BitsToSingle((uint)codes[index] << 16);
            }

            return;
        }

        for (; index < destination.Length; index++)
        {
            var code = BinaryPrimitives.ReadUInt16LittleEndian(source[(index * sizeof(ushort))..]);
            destination[index] = BitConverter.UInt32BitsToSingle((uint)code << 16);
        }
    }

    private static void WriteFp16(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (BitConverter.IsLittleEndian)
        {
            var codes = MemoryMarshal.Cast<byte, ushort>(source);
            for (var index = 0; index < codes.Length; index++)
            {
                destination[index] = (float)BitConverter.UInt16BitsToHalf(codes[index]);
            }

            return;
        }

        for (var index = 0; index < destination.Length; index++)
        {
            destination[index] = (float)BinaryPrimitives.ReadHalfLittleEndian(source[(index * sizeof(ushort))..]);
        }
    }

    private void RejectNonFinite(int index) => throw new SourceEncodingException(SourceEncodingFailure.NonFiniteValue,
        $"{EncodingId} block {index} fails validation: {SourceEncodingFailure.NonFiniteValue}.");
}
