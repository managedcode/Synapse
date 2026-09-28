using System.Buffers.Binary;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>
/// Scalar reference codec for <c>syn.ternary.absmean.g64.v1</c> row-major matrices.
/// </summary>
/// <remarks>
/// Each row is split into groups of 64 weights. A group stores a little-endian FP16
/// scale, <c>mean(|w|)</c> over the group's real elements rounded to the nearest FP16,
/// followed by 16 bytes of 2-bit codes: <c>00</c> is 0, <c>01</c> is +1, <c>10</c> is -1,
/// and <c>11</c> is reserved. Element <c>i</c> occupies bits <c>2*(i%4)</c> of byte
/// <c>i/4</c>. Codes are <c>clamp(round_to_even(w / scale), -1, 1)</c>, the BitNet b1.58
/// absmean rule applied per group. This post-training form is lossy by design;
/// sensitivity measurement decides where it may be used.
/// </remarks>
public static class TernaryBlockCodec
{
    /// <summary>Stable encoding identity.</summary>
    public const string EncodingId = "syn.ternary.absmean.g64.v1";

    /// <summary>Weights per group.</summary>
    public const int GroupElements = 64;

    /// <summary>Stored bytes per group: a two-byte scale and 16 code bytes.</summary>
    public const int GroupBytes = GroupedRowLayout.ScaleBytes + (GroupElements / 4);

    private const int ReservedCode = 0b11;
    private const int PositiveCode = 0b01;
    private const int NegativeCode = 0b10;

    /// <summary>Returns the exact stored byte length of a matrix.</summary>
    public static long GetEncodedLength(int rows, int columns) =>
        GroupedRowLayout.GetEncodedLength(rows, columns, GroupElements, GroupBytes);

    /// <summary>Returns stored bits per logical weight, including scales and padding.</summary>
    public static double GetStoredBitsPerWeight(int rows, int columns) =>
        GetEncodedLength(rows, columns) * 8.0 / ((long)rows * columns);

    /// <summary>Quantizes a row-major FP32 matrix after validating every source weight.</summary>
    public static void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination)
    {
        GroupedRowLayout.ValidateLengths(weights.Length, destination.Length, rows, columns, GroupElements, GroupBytes, EncodingId);
        GroupedRowLayout.ValidateFinite(weights, EncodingId);
        var groupsPerRow = GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        var scales = new Half[rows * groupsPerRow];
        for (var block = 0; block < scales.Length; block++)
        {
            scales[block] = ComputeScale(GetGroup(weights, columns, groupsPerRow, block));
        }

        for (var block = 0; block < scales.Length; block++)
        {
            WriteGroup(
                GetGroup(weights, columns, groupsPerRow, block),
                scales[block],
                destination.Slice(block * GroupBytes, GroupBytes));
        }
    }

    /// <summary>Dequantizes a matrix after validating every stored group.</summary>
    public static void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination)
    {
        GroupedRowLayout.ValidateLengths(destination.Length, encoded.Length, rows, columns, GroupElements, GroupBytes, EncodingId);
        ValidateEncoded(encoded, rows, columns);
        var groupsPerRow = GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        for (var block = 0; block < rows * groupsPerRow; block++)
        {
            var group = encoded.Slice(block * GroupBytes, GroupBytes);
            var scale = (float)GroupedRowLayout.ReadScale(group);
            var start = (block / groupsPerRow * columns) + (block % groupsPerRow * GroupElements);
            var count = GroupedRowLayout.GetGroupLength(columns, block % groupsPerRow, GroupElements);
            for (var index = 0; index < count; index++)
            {
                destination[start + index] = scale * ReadTrit(group, index);
            }
        }
    }

    /// <summary>Computes <c>output = W input</c> with additions only inside each group.</summary>
    public static void Multiply(
        ReadOnlySpan<byte> encoded,
        int rows,
        int columns,
        ReadOnlySpan<float> input,
        Span<float> output)
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
                sum += (double)GroupedRowLayout.ReadScale(block) * SumSelected(block, input, columns, group);
            }

            output[row] = (float)sum;
        }
    }

    private static double SumSelected(ReadOnlySpan<byte> block, ReadOnlySpan<float> input, int columns, int group)
    {
        var sum = 0.0;
        var count = GroupedRowLayout.GetGroupLength(columns, group, GroupElements);
        for (var index = 0; index < count; index++)
        {
            var value = (double)input[(group * GroupElements) + index];
            sum += ReadCode(block, index) switch
            {
                PositiveCode => value,
                NegativeCode => -value,
                _ => 0.0,
            };
        }

        return sum;
    }

    private static Half ComputeScale(ReadOnlySpan<float> group)
    {
        var sum = 0.0;
        foreach (var weight in group)
        {
            sum += MathF.Abs(weight);
        }

        var scale = (Half)(sum / group.Length);
        return Half.IsFinite(scale)
            ? scale
            : throw new QuantizationException(QuantizationFailure.ScaleOverflow, "Ternary group scale exceeds FP16 range.");
    }

    private static void WriteGroup(ReadOnlySpan<float> group, Half scale, Span<byte> block)
    {
        BinaryPrimitives.WriteHalfLittleEndian(block, scale);
        var codes = block[GroupedRowLayout.ScaleBytes..];
        codes.Clear();
        var divisor = (float)scale;
        if (divisor == 0f)
        {
            return;
        }

        for (var index = 0; index < group.Length; index++)
        {
            var trit = Math.Clamp((int)MathF.Round(group[index] / divisor, MidpointRounding.ToEven), -1, 1);
            var code = trit switch
            {
                1 => PositiveCode,
                -1 => NegativeCode,
                _ => 0,
            };
            codes[index >> 2] |= (byte)(code << ((index & 3) * 2));
        }
    }

    private static void ValidateEncoded(ReadOnlySpan<byte> encoded, int rows, int columns)
    {
        var groupsPerRow = GroupedRowLayout.GetGroupsPerRow(columns, GroupElements);
        for (var block = 0; block < rows * groupsPerRow; block++)
        {
            var group = encoded.Slice(block * GroupBytes, GroupBytes);
            var zeroScale = GroupedRowLayout.ReadValidScale(group, block, EncodingId) == Half.Zero;
            var validCount = GroupedRowLayout.GetGroupLength(columns, block % groupsPerRow, GroupElements);
            for (var index = 0; index < GroupElements; index++)
            {
                var code = ReadCode(group, index);
                if (code == ReservedCode)
                {
                    throw new QuantizationException(QuantizationFailure.ReservedCode, $"{EncodingId} group {block} uses the reserved code.");
                }

                if (code != 0 && (index >= validCount || zeroScale))
                {
                    throw new QuantizationException(
                        QuantizationFailure.NonCanonicalPadding,
                        $"{EncodingId} group {block} stores a non-zero padding or zero-scale code.");
                }
            }
        }
    }

    private static ReadOnlySpan<float> GetGroup(ReadOnlySpan<float> weights, int columns, int groupsPerRow, int block)
    {
        var group = block % groupsPerRow;
        var start = (block / groupsPerRow * columns) + (group * GroupElements);
        return weights.Slice(start, GroupedRowLayout.GetGroupLength(columns, group, GroupElements));
    }

    private static int ReadTrit(ReadOnlySpan<byte> block, int index) => ReadCode(block, index) switch
    {
        PositiveCode => 1,
        NegativeCode => -1,
        _ => 0,
    };

    private static int ReadCode(ReadOnlySpan<byte> block, int index) =>
        (block[GroupedRowLayout.ScaleBytes + (index >> 2)] >> ((index & 3) * 2)) & 0b11;
}
