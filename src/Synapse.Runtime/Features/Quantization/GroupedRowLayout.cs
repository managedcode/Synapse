using System.Buffers.Binary;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>Shared row-major layout rules for grouped codecs with one FP16 scale per group.</summary>
internal static class GroupedRowLayout
{
    public const int ScaleBytes = 2;

    public static int GetGroupsPerRow(int columns, int groupElements) => (columns + groupElements - 1) / groupElements;

    public static int GetGroupLength(int columns, int group, int groupElements) =>
        Math.Min(groupElements, columns - (group * groupElements));

    public static long GetEncodedLength(int rows, int columns, int groupElements, int groupBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        return checked((long)rows * GetGroupsPerRow(columns, groupElements) * groupBytes);
    }

    public static int GetGroupStart(int block, int columns, int groupElements)
    {
        var groupsPerRow = GetGroupsPerRow(columns, groupElements);
        return (block / groupsPerRow * columns) + (block % groupsPerRow * groupElements);
    }

    public static void ValidateLengths(
        int logicalLength,
        int encodedLength,
        int rows,
        int columns,
        int groupElements,
        int groupBytes,
        string encodingId)
    {
        var expectedEncoded = GetEncodedLength(rows, columns, groupElements, groupBytes);
        if (logicalLength != checked(rows * columns) || encodedLength != expectedEncoded)
        {
            throw new ArgumentException(
                $"{encodingId} matrix [{rows},{columns}] needs {rows * columns} weights and {expectedEncoded} bytes.");
        }
    }

    public static void ValidateLinearShape(int inputLength, int outputLength, int rows, int columns, string encodingId)
    {
        if (inputLength != columns || outputLength != rows)
        {
            throw new ArgumentException(
                $"{encodingId} linear expects [{columns}] -> [{rows}], got [{inputLength}] -> [{outputLength}].");
        }
    }

    public static void ValidateFinite(ReadOnlySpan<float> weights, string encodingId)
    {
        foreach (var weight in weights)
        {
            if (!float.IsFinite(weight))
            {
                throw new QuantizationException(
                    QuantizationFailure.NonFiniteWeight,
                    $"{encodingId} source contains a non-finite weight.");
            }
        }
    }

    public static Half ReadScale(ReadOnlySpan<byte> block) => BinaryPrimitives.ReadHalfLittleEndian(block);

    public static Half ReadValidScale(ReadOnlySpan<byte> block, int blockIndex, string encodingId)
    {
        var scale = ReadScale(block);
        return Half.IsFinite(scale) && !Half.IsNegative(scale)
            ? scale
            : throw new QuantizationException(
                QuantizationFailure.InvalidScale,
                $"{encodingId} group {blockIndex} has an invalid scale.");
    }
}
