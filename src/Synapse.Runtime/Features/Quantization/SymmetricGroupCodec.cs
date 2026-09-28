using System.Buffers.Binary;
using System.Globalization;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>
/// Scalar reference codec family <c>syn.q{bits}.symmetric.g{group}.v1</c> for any width
/// from 2 to 8 bits and any group size that is a multiple of 8 up to 1024.
/// </summary>
/// <remarks>
/// A group stores a little-endian FP16 scale followed by <c>group * bits / 8</c> bytes of
/// codes packed least-significant-bit first. Each code is <c>q + 2^(bits-1)</c> with
/// <c>q</c> in <c>[-(2^(bits-1) - 1), 2^(bits-1) - 1]</c>; code 0 is reserved. The scale is
/// <c>max(|w|) / (2^(bits-1) - 1)</c> rounded up to the next FP16 value, so no code is
/// clamped and the round-trip error is at most half a scale step. Codes round ties to
/// even. Padding and zero groups store <c>q = 0</c>. The 4-bit, 64-element member is
/// byte-identical to <c>syn.q4.symmetric.g64.v1</c> from spec §6.1.
/// </remarks>
public sealed class SymmetricGroupCodec : IWeightCodec
{
    /// <summary>Smallest supported code width.</summary>
    public const int MinimumBits = 2;

    /// <summary>Largest supported code width.</summary>
    public const int MaximumBits = 8;

    /// <summary>Largest supported group size.</summary>
    public const int MaximumGroupElements = 1024;

    private const int GroupAlignment = 8;

    private SymmetricGroupCodec(int bits, int groupElements)
    {
        Bits = bits;
        GroupElements = groupElements;
        GroupBytes = GroupedRowLayout.ScaleBytes + (groupElements * bits / 8);
        MaximumCode = (1 << (bits - 1)) - 1;
        CodeBias = 1 << (bits - 1);
        EncodingId = string.Create(CultureInfo.InvariantCulture, $"syn.q{bits}.symmetric.g{groupElements}.v1");
    }

    /// <summary>Code width in bits.</summary>
    public int Bits { get; }

    /// <summary>Weights per group.</summary>
    public int GroupElements { get; }

    /// <summary>Stored bytes per group, including the scale.</summary>
    public int GroupBytes { get; }

    /// <inheritdoc />
    public string EncodingId { get; }

    private int MaximumCode { get; }

    private int CodeBias { get; }

    /// <summary>Creates the codec for one width and group size.</summary>
    public static SymmetricGroupCodec Create(int bits, int groupElements)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(bits, MinimumBits);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(bits, MaximumBits);
        ArgumentOutOfRangeException.ThrowIfLessThan(groupElements, GroupAlignment);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(groupElements, MaximumGroupElements);
        if (groupElements % GroupAlignment != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(groupElements), "Group size must be a multiple of 8.");
        }

        return new SymmetricGroupCodec(bits, groupElements);
    }

    /// <inheritdoc />
    public long GetEncodedLength(int rows, int columns) =>
        GroupedRowLayout.GetEncodedLength(rows, columns, GroupElements, GroupBytes);

    /// <summary>Returns stored bits per logical weight, including scales and padding.</summary>
    public double GetStoredBitsPerWeight(int rows, int columns) =>
        GetEncodedLength(rows, columns) * 8.0 / ((long)rows * columns);

    /// <inheritdoc />
    public void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination)
    {
        GroupedRowLayout.ValidateLengths(weights.Length, destination.Length, rows, columns, GroupElements, GroupBytes, EncodingId);
        GroupedRowLayout.ValidateFinite(weights, EncodingId);
        var blocks = rows * GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        var scales = new Half[blocks];
        for (var block = 0; block < blocks; block++)
        {
            scales[block] = ComputeScale(GetGroup(weights, columns, block));
        }

        for (var block = 0; block < blocks; block++)
        {
            WriteGroup(GetGroup(weights, columns, block), scales[block], destination.Slice(block * GroupBytes, GroupBytes));
        }
    }

    /// <inheritdoc />
    public void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination)
    {
        GroupedRowLayout.ValidateLengths(destination.Length, encoded.Length, rows, columns, GroupElements, GroupBytes, EncodingId);
        ValidateEncoded(encoded, rows, columns);
        var blocks = rows * GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        for (var block = 0; block < blocks; block++)
        {
            var group = encoded.Slice(block * GroupBytes, GroupBytes);
            var scale = (float)GroupedRowLayout.ReadScale(group);
            var start = GroupedRowLayout.GetGroupStart(block, columns, GroupElements);
            var count = GetGroupLength(columns, block);
            for (var index = 0; index < count; index++)
            {
                destination[start + index] = scale * (ReadCode(group, index) - CodeBias);
            }
        }
    }

    /// <inheritdoc />
    public void Multiply(ReadOnlySpan<byte> encoded, int rows, int columns, ReadOnlySpan<float> input, Span<float> output)
    {
        GroupedRowLayout.ValidateLengths(rows * columns, encoded.Length, rows, columns, GroupElements, GroupBytes, EncodingId);
        GroupedRowLayout.ValidateLinearShape(input.Length, output.Length, rows, columns, EncodingId);
        ValidateEncoded(encoded, rows, columns);
        var groupsPerRow = GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        for (var row = 0; row < rows; row++)
        {
            var sum = 0.0;
            for (var group = 0; group < groupsPerRow; group++)
            {
                var block = encoded.Slice(((row * groupsPerRow) + group) * GroupBytes, GroupBytes);
                var groupSum = 0.0;
                var count = GroupedRowLayout.GetGroupLength(columns, group, GroupElements);
                for (var index = 0; index < count; index++)
                {
                    groupSum += (ReadCode(block, index) - CodeBias) * (double)input[(group * GroupElements) + index];
                }

                sum += (double)GroupedRowLayout.ReadScale(block) * groupSum;
            }

            output[row] = (float)sum;
        }
    }

    private Half ComputeScale(ReadOnlySpan<float> group)
    {
        var maximum = 0f;
        foreach (var weight in group)
        {
            maximum = MathF.Max(maximum, MathF.Abs(weight));
        }

        if (maximum == 0f)
        {
            return Half.Zero;
        }

        var exact = maximum / MaximumCode;
        var scale = (Half)exact;
        if ((float)scale < exact)
        {
            scale = Half.BitIncrement(scale);
        }

        return Half.IsFinite(scale)
            ? scale
            : throw new QuantizationException(QuantizationFailure.ScaleOverflow, $"{EncodingId} group scale exceeds FP16 range.");
    }

    private void WriteGroup(ReadOnlySpan<float> group, Half scale, Span<byte> block)
    {
        BinaryPrimitives.WriteHalfLittleEndian(block, scale);
        var divisor = (float)scale;
        for (var index = 0; index < GroupElements; index++)
        {
            var quantized = index < group.Length && divisor != 0f
                ? (int)MathF.Round(group[index] / divisor, MidpointRounding.ToEven)
                : 0;
            WriteCode(block, index, quantized + CodeBias);
        }
    }

    private void ValidateEncoded(ReadOnlySpan<byte> encoded, int rows, int columns)
    {
        var blocks = rows * GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        for (var block = 0; block < blocks; block++)
        {
            var group = encoded.Slice(block * GroupBytes, GroupBytes);
            var zeroScale = GroupedRowLayout.ReadValidScale(group, block, EncodingId) == Half.Zero;
            var validCount = GetGroupLength(columns, block);
            for (var index = 0; index < GroupElements; index++)
            {
                var code = ReadCode(group, index);
                if (code == 0)
                {
                    throw new QuantizationException(QuantizationFailure.ReservedCode, $"{EncodingId} group {block} uses the reserved code.");
                }

                if (code != CodeBias && (index >= validCount || zeroScale))
                {
                    throw new QuantizationException(
                        QuantizationFailure.NonCanonicalPadding,
                        $"{EncodingId} group {block} stores a non-zero padding or zero-scale code.");
                }
            }
        }
    }

    private ReadOnlySpan<float> GetGroup(ReadOnlySpan<float> weights, int columns, int block) =>
        weights.Slice(GroupedRowLayout.GetGroupStart(block, columns, GroupElements), GetGroupLength(columns, block));

    private int GetGroupLength(int columns, int block) =>
        GroupedRowLayout.GetGroupLength(columns, block % GroupedRowLayout.GetGroupsPerRow(columns, GroupElements), GroupElements);

    private int ReadCode(ReadOnlySpan<byte> block, int index)
    {
        var bitOffset = index * Bits;
        var byteIndex = GroupedRowLayout.ScaleBytes + (bitOffset >> 3);
        var shift = bitOffset & 7;
        int value = block[byteIndex];
        if (shift + Bits > 8)
        {
            value |= block[byteIndex + 1] << 8;
        }

        return (value >> shift) & ((1 << Bits) - 1);
    }

    private void WriteCode(Span<byte> block, int index, int code)
    {
        var bitOffset = index * Bits;
        var byteIndex = GroupedRowLayout.ScaleBytes + (bitOffset >> 3);
        var shift = bitOffset & 7;
        var mask = ((1 << Bits) - 1) << shift;
        var value = (code << shift) & mask;
        block[byteIndex] = (byte)((block[byteIndex] & ~mask) | value);
        if (shift + Bits > 8)
        {
            block[byteIndex + 1] = (byte)((block[byteIndex + 1] & ~(mask >> 8)) | (value >> 8));
        }
    }
}
