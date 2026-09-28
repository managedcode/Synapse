using System.Buffers.Binary;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class QuantizedBlockCodecTests
{
    private const byte ZeroPair = 0x88;

    [Test]
    public async Task Q4GoldenBytesMatch()
    {
        var weights = new float[SynQ4BlockCodec.GroupElements];
        float[] prefix = [7f, -7f, 3.5f, 2.5f, -2.5f, 0.5f, 1.5f];
        prefix.CopyTo(weights, 0);
        var encoded = new byte[SynQ4BlockCodec.GetEncodedLength(1, weights.Length)];

        SynQ4BlockCodec.Encode(weights, rows: 1, columns: weights.Length, encoded);

        byte[] expectedPrefix = [0x00, 0x3C, 0x1F, 0xAC, 0x86, 0x8A];
        await Assert.That(encoded.Take(expectedPrefix.Length)).IsEquivalentTo(expectedPrefix);
        await Assert.That(encoded.Skip(expectedPrefix.Length).All(value => value == ZeroPair)).IsTrue();
    }

    [Test]
    public async Task ZeroGroupHasZeroScaleAndZeroCodes()
    {
        var weights = new float[SynQ4BlockCodec.GroupElements];
        var encoded = new byte[SynQ4BlockCodec.GroupBytes];

        SynQ4BlockCodec.Encode(weights, rows: 1, columns: weights.Length, encoded);

        await Assert.That(encoded[0]).IsEqualTo((byte)0);
        await Assert.That(encoded[1]).IsEqualTo((byte)0);
        await Assert.That(encoded.Skip(2).All(value => value == ZeroPair)).IsTrue();
    }

    [Test]
    public async Task PaddingEncodesZeroAndIsDroppedOnDecode()
    {
        const int Columns = 65;
        var weights = Enumerable.Range(0, Columns).Select(index => index == 64 ? 3f : 0f).ToArray();
        var encoded = new byte[SynQ4BlockCodec.GetEncodedLength(1, Columns)];
        var decoded = new float[Columns];

        SynQ4BlockCodec.Encode(weights, rows: 1, columns: Columns, encoded);
        SynQ4BlockCodec.Decode(encoded, rows: 1, columns: Columns, decoded);

        var secondGroupCodes = encoded.AsSpan(SynQ4BlockCodec.GroupBytes + 2).ToArray();
        await Assert.That(secondGroupCodes[0] >> 4).IsEqualTo(8);
        await Assert.That(secondGroupCodes.Skip(1).All(value => value == ZeroPair)).IsTrue();
        await Assert.That(MathF.Abs(decoded[64] - 3f)).IsLessThan(0.01f);
    }

    [Test]
    public async Task RoundTripErrorIsBoundedByHalfScale()
    {
        var random = new Random(4211);
        foreach (var columns in new[] { 1, 63, 64, 65, 130, 257 })
        {
            const int Rows = 3;
            var weights = Enumerable.Range(0, Rows * columns)
                .Select(_ => (float)((random.NextDouble() * 20) - 10)).ToArray();
            var encoded = new byte[SynQ4BlockCodec.GetEncodedLength(Rows, columns)];
            var decoded = new float[weights.Length];

            SynQ4BlockCodec.Encode(weights, Rows, columns, encoded);
            SynQ4BlockCodec.Decode(encoded, Rows, columns, decoded);

            var groupsPerRow = (columns + SynQ4BlockCodec.GroupElements - 1) / SynQ4BlockCodec.GroupElements;
            for (var index = 0; index < weights.Length; index++)
            {
                var row = index / columns;
                var group = index % columns / SynQ4BlockCodec.GroupElements;
                var offset = ((row * groupsPerRow) + group) * SynQ4BlockCodec.GroupBytes;
                var scale = (float)BinaryPrimitives.ReadHalfLittleEndian(encoded.AsSpan(offset, 2));
                await Assert.That(MathF.Abs(weights[index] - decoded[index]))
                    .IsLessThanOrEqualTo((scale * 0.5f) + 1e-6f);
            }
        }
    }

    [Test]
    public async Task InvalidScaleRejected()
    {
        var encoded = new byte[SynQ4BlockCodec.GroupBytes];
        SynQ4BlockCodec.Encode(new float[SynQ4BlockCodec.GroupElements], 1, SynQ4BlockCodec.GroupElements, encoded);
        var decoded = new float[SynQ4BlockCodec.GroupElements];
        decoded.AsSpan().Fill(42f);

        foreach (var scaleBits in new ushort[] { 0x7E00, 0x7C00, 0xFC00, 0xBC00, 0x8000 })
        {
            BinaryPrimitives.WriteUInt16LittleEndian(encoded, scaleBits);
            var failure = await Assert.That(() => SynQ4BlockCodec.Decode(
                encoded, 1, SynQ4BlockCodec.GroupElements, decoded)).Throws<QuantizationException>();
            await Assert.That(failure!.Failure).IsEqualTo(QuantizationFailure.InvalidScale);
        }

        await Assert.That(decoded.All(value => value == 42f)).IsTrue();
    }

    [Test]
    public async Task ReservedCodeAndNonCanonicalPaddingRejected()
    {
        var weights = Enumerable.Range(0, 65).Select(index => (float)(index % 7)).ToArray();
        var encoded = new byte[SynQ4BlockCodec.GetEncodedLength(1, weights.Length)];
        SynQ4BlockCodec.Encode(weights, 1, weights.Length, encoded);
        var decoded = new float[weights.Length];

        var reserved = encoded.ToArray();
        reserved[2] = (byte)(reserved[2] & 0xF0);
        var reservedFailure = await Assert.That(() => SynQ4BlockCodec.Decode(
            reserved, 1, weights.Length, decoded)).Throws<QuantizationException>();

        var padding = encoded.ToArray();
        padding[^1] = 0x89;
        var paddingFailure = await Assert.That(() => SynQ4BlockCodec.Decode(
            padding, 1, weights.Length, decoded)).Throws<QuantizationException>();

        await Assert.That(reservedFailure!.Failure).IsEqualTo(QuantizationFailure.ReservedCode);
        await Assert.That(paddingFailure!.Failure).IsEqualTo(QuantizationFailure.NonCanonicalPadding);
    }

    [Test]
    public async Task NonFiniteAndOverflowingWeightsRejectedBeforeWriting()
    {
        var encoded = new byte[SynQ4BlockCodec.GroupBytes];
        encoded.AsSpan().Fill(0xAB);

        foreach (var (value, expected) in new[]
        {
            (float.NaN, QuantizationFailure.NonFiniteWeight),
            (float.PositiveInfinity, QuantizationFailure.NonFiniteWeight),
            (1_000_000f, QuantizationFailure.ScaleOverflow),
        })
        {
            var weights = new float[SynQ4BlockCodec.GroupElements];
            weights[5] = value;
            var failure = await Assert.That(() => SynQ4BlockCodec.Encode(
                weights, 1, weights.Length, encoded)).Throws<QuantizationException>();
            await Assert.That(failure!.Failure).IsEqualTo(expected);
        }

        await Assert.That(encoded.All(value => value == 0xAB)).IsTrue();
    }

    [Test]
    public async Task SizeAccountingExact()
    {
        await Assert.That(SynQ4BlockCodec.GetEncodedLength(1, 64)).IsEqualTo(34L);
        await Assert.That(SynQ4BlockCodec.GetEncodedLength(1, 65)).IsEqualTo(68L);
        await Assert.That(SynQ4BlockCodec.GetEncodedLength(3, 128)).IsEqualTo(204L);
        await Assert.That(SynQ4BlockCodec.GetStoredBitsPerWeight(1, 64)).IsEqualTo(4.25);
        await Assert.That(SynQ4BlockCodec.GetStoredBitsPerWeight(1, 1)).IsEqualTo(272.0);
        await Assert.That(() => SynQ4BlockCodec.GetEncodedLength(0, 64)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task Q4LinearMatchesDequantizedReference()
    {
        var random = new Random(733);
        const int Rows = 5;
        const int Columns = 130;
        var weights = Enumerable.Range(0, Rows * Columns)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var input = Enumerable.Range(0, Columns)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var encoded = new byte[SynQ4BlockCodec.GetEncodedLength(Rows, Columns)];
        var decoded = new float[weights.Length];
        var output = new float[Rows];
        SynQ4BlockCodec.Encode(weights, Rows, Columns, encoded);
        SynQ4BlockCodec.Decode(encoded, Rows, Columns, decoded);

        SynQ4BlockCodec.Multiply(encoded, Rows, Columns, input, output);

        for (var row = 0; row < Rows; row++)
        {
            var expected = 0.0;
            for (var column = 0; column < Columns; column++)
            {
                expected += (double)decoded[(row * Columns) + column] * input[column];
            }

            await Assert.That(Math.Abs(output[row] - expected)).IsLessThan(1e-4);
        }
    }
}
