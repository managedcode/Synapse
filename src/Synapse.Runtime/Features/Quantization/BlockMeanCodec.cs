using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.InteropServices;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>
/// Experimental lossy row-major codec that replaces adjacent weights with their FP32 mean.
/// This changes the linear operator and requires independent model quality qualification.
/// </summary>
public sealed class BlockMeanCodec : IWeightCodec
{
    /// <summary>Largest supported approximation group.</summary>
    public const int MaximumGroupElements = 1024;

    private BlockMeanCodec(int groupElements)
    {
        GroupElements = groupElements;
        EncodingId = string.Create(CultureInfo.InvariantCulture, $"syn.approx.blockmean.g{groupElements}.f32.v1");
    }

    /// <summary>Adjacent logical weights replaced by one mean.</summary>
    public int GroupElements { get; }

    /// <inheritdoc />
    public string EncodingId { get; }

    /// <summary>Creates an explicitly approximate codec; the default group has four weights.</summary>
    public static BlockMeanCodec Create(int groupElements = 4)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(groupElements, 2);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(groupElements, MaximumGroupElements);
        return new BlockMeanCodec(groupElements);
    }

    /// <summary>Number of stored means in each row, including a possible partial final group.</summary>
    public int GetGroupsPerRow(int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        return ((columns - 1) / GroupElements) + 1;
    }

    /// <inheritdoc />
    public long GetEncodedLength(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        return checked((long)rows * GetGroupsPerRow(columns) * sizeof(float));
    }

    /// <inheritdoc />
    /// <remarks>Source and destination must not share storage; rejection leaves destination unchanged.</remarks>
    public void Encode(ReadOnlySpan<float> weights, int rows, int columns, Span<byte> destination)
    {
        ValidateMatrixLengths(weights.Length, destination.Length, rows, columns);
        RejectOverlap(MemoryMarshal.AsBytes(weights), destination);
        GroupedRowLayout.ValidateFinite(weights, EncodingId);
        var groups = GetGroupsPerRow(columns);
        for (var row = 0; row < rows; row++)
        {
            for (var group = 0; group < groups; group++)
            {
                var start = group * GroupElements;
                var count = Math.Min(GroupElements, columns - start);
                var sum = 0.0;
                foreach (var weight in weights.Slice((row * columns) + start, count))
                {
                    sum += weight;
                }

                BinaryPrimitives.WriteSingleLittleEndian(destination[(((row * groups) + group) * sizeof(float))..], (float)(sum / count));
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>Encoded and destination storage must be disjoint. Every mean is validated before writing.</remarks>
    public void Decode(ReadOnlySpan<byte> encoded, int rows, int columns, Span<float> destination)
    {
        ValidateMatrixLengths(destination.Length, encoded.Length, rows, columns);
        RejectOverlap(encoded, MemoryMarshal.AsBytes(destination));
        ValidateMeans(encoded);
        var groups = GetGroupsPerRow(columns);
        for (var row = 0; row < rows; row++)
        {
            for (var group = 0; group < groups; group++)
            {
                var start = group * GroupElements;
                var count = Math.Min(GroupElements, columns - start);
                var mean = BinaryPrimitives.ReadSingleLittleEndian(encoded[(((row * groups) + group) * sizeof(float))..]);
                destination.Slice((row * columns) + start, count).Fill(mean);
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Rents scratch buffers and publishes only complete finite results. Output may overlap input or encoded bytes.
    /// FP64 grouped/SIMD accumulation may differ in rounding from a dense accumulation of expanded weights.
    /// </remarks>
    public void Multiply(ReadOnlySpan<byte> encoded, int rows, int columns, ReadOnlySpan<float> input, Span<float> output)
    {
        ValidateLinearLengths(encoded.Length, rows, columns, input.Length, output.Length);
        var groups = GetGroupsPerRow(columns);
        var sums = ArrayPool<double>.Shared.Rent(groups);
        var results = ArrayPool<float>.Shared.Rent(rows);
        try
        {
            Multiply(encoded, rows, columns, input, output, sums.AsSpan(0, groups), results.AsSpan(0, rows));
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sums);
            ArrayPool<float>.Shared.Return(results);
        }
    }

    /// <summary>Computes direct linear with caller-owned reusable buffers and no managed allocations.</summary>
    /// <param name="encoded">One finite little-endian FP32 mean per logical group.</param>
    /// <param name="rows">Logical matrix row count.</param>
    /// <param name="columns">Logical matrix column count.</param>
    /// <param name="input">Finite logical input vector.</param>
    /// <param name="output">Destination, published only after every row is finite.</param>
    /// <param name="groupSums">At least <see cref="GetGroupsPerRow"/> entries; overwritten.</param>
    /// <param name="rowScratch">At least <paramref name="rows"/> entries; overwritten.</param>
    /// <remarks>Used scratch ranges must be disjoint from each other, inputs, encoded bytes, and output.</remarks>
    public void Multiply(
        ReadOnlySpan<byte> encoded,
        int rows,
        int columns,
        ReadOnlySpan<float> input,
        Span<float> output,
        Span<double> groupSums,
        Span<float> rowScratch)
    {
        ValidateLinearLengths(encoded.Length, rows, columns, input.Length, output.Length);
        var groups = GetGroupsPerRow(columns);
        if (groupSums.Length < groups || rowScratch.Length < rows)
        {
            throw new ArgumentException("Block-mean linear scratch is smaller than the matrix requires.");
        }

        groupSums = groupSums[..groups];
        rowScratch = rowScratch[..rows];
        ValidateScratch(encoded, input, output, groupSums, rowScratch);
        ValidateMeans(encoded);
        foreach (var value in input)
        {
            if (!float.IsFinite(value))
            {
                throw new ArgumentException("Block-mean linear input must be finite.", nameof(input));
            }
        }

        BlockMeanMath.SumGroups(input, GroupElements, groupSums);
        for (var row = 0; row < rows; row++)
        {
            var means = encoded.Slice(row * groups * sizeof(float), groups * sizeof(float));
            var value = (float)BlockMeanMath.Dot(means, groupSums);
            if (!float.IsFinite(value))
            {
                throw new ArithmeticException("Block-mean linear output exceeds finite FP32 range.");
            }

            rowScratch[row] = value;
        }

        rowScratch.CopyTo(output);
    }

    private void ValidateMatrixLengths(int logicalLength, int encodedLength, int rows, int columns)
    {
        var expectedEncoded = GetEncodedLength(rows, columns);
        if (logicalLength != checked(rows * columns) || encodedLength != expectedEncoded)
        {
            throw new ArgumentException($"{EncodingId} matrix shape or encoded length does not match [{rows},{columns}].");
        }
    }

    private void ValidateLinearLengths(int encodedLength, int rows, int columns, int inputLength, int outputLength)
    {
        if (encodedLength != GetEncodedLength(rows, columns) || inputLength != columns || outputLength != rows)
        {
            throw new ArgumentException($"{EncodingId} linear shape or encoded length does not match [{rows},{columns}].");
        }
    }

    private static void ValidateMeans(ReadOnlySpan<byte> encoded)
    {
        for (var offset = 0; offset < encoded.Length; offset += sizeof(float))
        {
            if (!float.IsFinite(BinaryPrimitives.ReadSingleLittleEndian(encoded[offset..])))
            {
                throw new FormatException("Block-mean encoded data contains a non-finite mean.");
            }
        }
    }

    private static void ValidateScratch(
        ReadOnlySpan<byte> encoded,
        ReadOnlySpan<float> input,
        Span<float> output,
        Span<double> sums,
        Span<float> rows)
    {
        var sumBytes = MemoryMarshal.AsBytes(sums);
        var rowBytes = MemoryMarshal.AsBytes(rows);
        RejectOverlap(sumBytes, rowBytes);
        RejectOverlap(sumBytes, encoded);
        RejectOverlap(sumBytes, MemoryMarshal.AsBytes(input));
        RejectOverlap(sumBytes, MemoryMarshal.AsBytes(output));
        RejectOverlap(rowBytes, encoded);
        RejectOverlap(rowBytes, MemoryMarshal.AsBytes(input));
        RejectOverlap(rowBytes, MemoryMarshal.AsBytes(output));
    }

    private static void RejectOverlap(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right)
    {
        if (left.Overlaps(right))
        {
            throw new ArgumentException("Block-mean source, destination, or scratch storage overlaps an unsupported range.");
        }
    }
}
