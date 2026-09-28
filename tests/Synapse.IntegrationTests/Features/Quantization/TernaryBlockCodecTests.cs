using System.Buffers.Binary;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class TernaryBlockCodecTests
{
    [Test]
    public async Task TernaryGoldenBytesMatch()
    {
        var weights = new float[TernaryBlockCodec.GroupElements];
        float[] prefix = [1f, -1f, 0.9375f, -0.9375f, 0.03125f, 0.09375f];
        prefix.CopyTo(weights, 0);
        var encoded = new byte[TernaryBlockCodec.GetEncodedLength(1, weights.Length)];

        TernaryBlockCodec.Encode(weights, 1, weights.Length, encoded);

        byte[] expectedPrefix = [0x00, 0x2C, 0x99, 0x04];
        await Assert.That(encoded.Take(expectedPrefix.Length)).IsEquivalentTo(expectedPrefix);
        await Assert.That(encoded.Skip(expectedPrefix.Length).All(value => value == 0)).IsTrue();
    }

    [Test]
    public async Task TernaryDecodesToSignedGroupScaleOrZero()
    {
        var random = new Random(9001);
        const int Rows = 4;
        const int Columns = 150;
        var weights = Enumerable.Range(0, Rows * Columns)
            .Select(_ => (float)((random.NextDouble() * 4) - 2)).ToArray();
        var encoded = new byte[TernaryBlockCodec.GetEncodedLength(Rows, Columns)];
        var decoded = new float[weights.Length];

        TernaryBlockCodec.Encode(weights, Rows, Columns, encoded);
        TernaryBlockCodec.Decode(encoded, Rows, Columns, decoded);

        const int GroupsPerRow = 3;
        for (var index = 0; index < weights.Length; index++)
        {
            var block = (index / Columns * GroupsPerRow) + (index % Columns / TernaryBlockCodec.GroupElements);
            var scale = (float)BinaryPrimitives.ReadHalfLittleEndian(
                encoded.AsSpan(block * TernaryBlockCodec.GroupBytes, 2));
            var magnitude = MathF.Abs(decoded[index]);
            await Assert.That(magnitude == 0f || magnitude == scale).IsTrue();
            await Assert.That(decoded[index] == 0f || MathF.Sign(decoded[index]) == MathF.Sign(weights[index])).IsTrue();
        }
    }

    [Test]
    public async Task TernaryLinearMatchesDequantizedReference()
    {
        var random = new Random(12);
        const int Rows = 6;
        const int Columns = 97;
        var weights = Enumerable.Range(0, Rows * Columns)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var input = Enumerable.Range(0, Columns)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var encoded = new byte[TernaryBlockCodec.GetEncodedLength(Rows, Columns)];
        var decoded = new float[weights.Length];
        var output = new float[Rows];
        TernaryBlockCodec.Encode(weights, Rows, Columns, encoded);
        TernaryBlockCodec.Decode(encoded, Rows, Columns, decoded);

        TernaryBlockCodec.Multiply(encoded, Rows, Columns, input, output);

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

    [Test]
    public async Task TernaryRejectsReservedCodeInvalidScaleAndPadding()
    {
        var weights = Enumerable.Range(0, 65).Select(index => (float)((index % 3) - 1)).ToArray();
        var encoded = new byte[TernaryBlockCodec.GetEncodedLength(1, weights.Length)];
        TernaryBlockCodec.Encode(weights, 1, weights.Length, encoded);
        var decoded = new float[weights.Length];

        var reserved = encoded.ToArray();
        reserved[2] = (byte)(reserved[2] | 0x03);
        var invalidScale = encoded.ToArray();
        BinaryPrimitives.WriteUInt16LittleEndian(invalidScale, 0x7E00);
        var padding = encoded.ToArray();
        padding[^1] = 0x40;

        var failures = new List<QuantizationFailure>();
        foreach (var candidate in new[] { reserved, invalidScale, padding })
        {
            var failure = await Assert.That(() => TernaryBlockCodec.Decode(
                candidate, 1, weights.Length, decoded)).Throws<QuantizationException>();
            failures.Add(failure!.Failure);
        }

        await Assert.That(failures).IsEquivalentTo(
            [QuantizationFailure.ReservedCode, QuantizationFailure.InvalidScale, QuantizationFailure.NonCanonicalPadding]);
    }

    [Test]
    public async Task TernarySizeAccountingExact()
    {
        await Assert.That(TernaryBlockCodec.GetEncodedLength(1, 64)).IsEqualTo(18L);
        await Assert.That(TernaryBlockCodec.GetEncodedLength(2, 65)).IsEqualTo(72L);
        await Assert.That(TernaryBlockCodec.GetStoredBitsPerWeight(1, 64)).IsEqualTo(2.25);
    }
}
