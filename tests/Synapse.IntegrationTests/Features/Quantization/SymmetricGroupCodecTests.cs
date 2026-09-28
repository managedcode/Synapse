using System.Buffers.Binary;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class SymmetricGroupCodecTests
{
    [Test]
    public async Task EncodingIdNamesBitsAndGroup()
    {
        await Assert.That(SymmetricGroupCodec.Create(6, 32).EncodingId).IsEqualTo("syn.q6.symmetric.g32.v1");
        await Assert.That(SymmetricGroupCodec.Create(4, 64).EncodingId).IsEqualTo(SynQ4BlockCodec.EncodingId);
        await Assert.That(SymmetricGroupCodec.Create(8, 128).GroupBytes).IsEqualTo(130);
        await Assert.That(SymmetricGroupCodec.Create(3, 64).GroupBytes).IsEqualTo(26);
    }

    [Test]
    public async Task OddWidthGoldenBytesMatch()
    {
        var codec = SymmetricGroupCodec.Create(3, 8);
        float[] weights = [3f, -3f, 1f, 0f, -1f, 2f, -2f, 0.5f];
        var encoded = new byte[codec.GetEncodedLength(1, weights.Length)];

        codec.Encode(weights, 1, weights.Length, encoded);

        await Assert.That(encoded).IsEquivalentTo(new byte[] { 0x00, 0x3C, 0x4F, 0x39, 0x8B });
    }

    [Test]
    public async Task EveryBitWidthRoundTripsWithinHalfScale()
    {
        var random = new Random(515);
        for (var bits = 2; bits <= 8; bits++)
        {
            foreach (var group in new[] { 8, 32, 64, 128 })
            {
                var codec = SymmetricGroupCodec.Create(bits, group);
                foreach (var columns in new[] { 1, 33, 130 })
                {
                    const int Rows = 2;
                    var weights = Enumerable.Range(0, Rows * columns)
                        .Select(_ => (float)((random.NextDouble() * 6) - 3)).ToArray();
                    var encoded = new byte[codec.GetEncodedLength(Rows, columns)];
                    var decoded = new float[weights.Length];

                    codec.Encode(weights, Rows, columns, encoded);
                    codec.Decode(encoded, Rows, columns, decoded);

                    var groupsPerRow = (columns + group - 1) / group;
                    await Assert.That(encoded.Length).IsEqualTo(Rows * groupsPerRow * codec.GroupBytes);
                    await Assert.That(MaximumExcess(weights, decoded, encoded, columns, group, codec.GroupBytes)).IsLessThanOrEqualTo(1e-6f);
                }
            }
        }
    }

    [Test]
    public async Task MoreBitsNeverIncreaseDistortion()
    {
        var random = new Random(8080);
        var weights = Enumerable.Range(0, 16 * 256).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var calibration = Enumerable.Range(0, 8 * 256).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();

        var distortions = Enumerable.Range(2, 7)
            .Select(bits => WeightSensitivity.MeasureCodec(SymmetricGroupCodec.Create(bits, 64), weights, 16, 256, calibration).Relative)
            .ToArray();

        for (var index = 1; index < distortions.Length; index++)
        {
            await Assert.That(distortions[index]).IsLessThan(distortions[index - 1]);
        }
    }

    [Test]
    public async Task EveryBitWidthLinearMatchesDequantizedReference()
    {
        var random = new Random(31);
        const int Rows = 3;
        const int Columns = 70;
        var weights = Enumerable.Range(0, Rows * Columns).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var input = Enumerable.Range(0, Columns).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        for (var bits = 2; bits <= 8; bits++)
        {
            var codec = SymmetricGroupCodec.Create(bits, 32);
            var encoded = new byte[codec.GetEncodedLength(Rows, Columns)];
            var decoded = new float[weights.Length];
            var output = new float[Rows];
            codec.Encode(weights, Rows, Columns, encoded);
            codec.Decode(encoded, Rows, Columns, decoded);

            codec.Multiply(encoded, Rows, Columns, input, output);

            for (var row = 0; row < Rows; row++)
            {
                var expected = Enumerable.Range(0, Columns).Sum(column => (double)decoded[(row * Columns) + column] * input[column]);
                await Assert.That(Math.Abs(output[row] - expected)).IsLessThan(1e-4);
            }
        }
    }

    [Test]
    public async Task ReservedCodeRejectedForAnyWidth()
    {
        var codec = SymmetricGroupCodec.Create(5, 32);
        var weights = Enumerable.Range(0, 32).Select(index => (index % 9) - 4f).ToArray();
        var encoded = new byte[codec.GetEncodedLength(1, weights.Length)];
        codec.Encode(weights, 1, weights.Length, encoded);
        encoded[2] = (byte)(encoded[2] & 0xE0);

        var failure = await Assert.That(() => codec.Decode(encoded, 1, weights.Length, new float[weights.Length]))
            .Throws<QuantizationException>();

        await Assert.That(failure!.Failure).IsEqualTo(QuantizationFailure.ReservedCode);
    }

    [Test]
    public async Task InvalidParametersRejected()
    {
        foreach (var (bits, group) in new[] { (1, 64), (9, 64), (4, 0), (4, 12), (4, 2048) })
        {
            await Assert.That(() => SymmetricGroupCodec.Create(bits, group)).Throws<ArgumentOutOfRangeException>();
        }
    }

    [Test]
    public async Task WeightCodecsResolveById()
    {
        var resolved = WeightCodecs.TryGet("syn.q5.symmetric.g128.v1", out var codec);
        var ternary = WeightCodecs.TryGet(TernaryBlockCodec.EncodingId, out var ternaryCodec);

        await Assert.That(resolved).IsTrue();
        await Assert.That(codec!.EncodingId).IsEqualTo("syn.q5.symmetric.g128.v1");
        await Assert.That(ternary).IsTrue();
        await Assert.That(ternaryCodec!.EncodingId).IsEqualTo(TernaryBlockCodec.EncodingId);
        await Assert.That(WeightCodecs.TryGet("syn.q9.symmetric.g64.v1", out _)).IsFalse();
        await Assert.That(WeightCodecs.TryGet("gguf.q4_k", out _)).IsFalse();
    }

    private static float MaximumExcess(float[] weights, float[] decoded, byte[] encoded, int columns, int group, int groupBytes)
    {
        var groupsPerRow = (columns + group - 1) / group;
        var excess = 0f;
        for (var index = 0; index < weights.Length; index++)
        {
            var block = (index / columns * groupsPerRow) + (index % columns / group);
            var scale = (float)BinaryPrimitives.ReadHalfLittleEndian(encoded.AsSpan(block * groupBytes, 2));
            excess = MathF.Max(excess, MathF.Abs(weights[index] - decoded[index]) - (scale * 0.5f));
        }

        return excess;
    }
}
