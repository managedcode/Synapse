using System.Buffers.Binary;
using System.Runtime.InteropServices;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class BlockMeanCodecTests
{
    [Test]
    public async Task BlockMeanCodecIsAnExplicitExperiment()
    {
        var type = Type.GetType("ManagedCode.Synapse.Runtime.Features.Quantization.BlockMeanCodec, ManagedCode.Synapse.Runtime");

        await Assert.That(type).IsNotNull();
        await Assert.That(BlockMeanCodec.Create().EncodingId).IsEqualTo("syn.approx.blockmean.g4.f32.v1");
        await Assert.That(WeightCodecs.TryGet(BlockMeanCodec.Create().EncodingId, out _)).IsFalse();
    }

    [Test]
    public async Task GoldenMeansAndPartialGroupsMatch()
    {
        float[] weights = [1, 2, 3, 4, 9, -1, -2, -3, -4, 7];
        var codec = BlockMeanCodec.Create();
        var encoded = new byte[codec.GetEncodedLength(2, 5)];
        var decoded = new float[weights.Length];

        codec.Encode(weights, 2, 5, encoded);
        codec.Decode(encoded, 2, 5, decoded);

        byte[] expected =
        [
            0, 0, 0x20, 0x40, 0, 0, 0x10, 0x41,
            0, 0, 0x20, 0xC0, 0, 0, 0xE0, 0x40,
        ];
        float[] expectedWeights = [2.5f, 2.5f, 2.5f, 2.5f, 9, -2.5f, -2.5f, -2.5f, -2.5f, 7];
        await Assert.That(encoded).IsEquivalentTo(expected);
        await Assert.That(decoded).IsEquivalentTo(expectedWeights);
    }

    [Test]
    public async Task ExactSizesAndInvalidShapesAreChecked()
    {
        var codec = BlockMeanCodec.Create();

        await Assert.That(codec.GetEncodedLength(3, 5)).IsEqualTo(24L);
        await Assert.That(codec.GetEncodedLength(1, int.MaxValue)).IsEqualTo(2_147_483_648L);
        await Assert.That(codec.GetGroupsPerRow(1)).IsEqualTo(1);
        foreach (var group in new[] { 0, 1, 1025 })
        {
            await Assert.That(() => BlockMeanCodec.Create(group)).Throws<ArgumentOutOfRangeException>();
        }

        await Assert.That(() => codec.GetEncodedLength(0, 4)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => codec.Encode(new float[4], 1, 4, new byte[3])).Throws<ArgumentException>();
        await Assert.That(() => codec.Encode([], int.MaxValue, 4, [])).Throws<OverflowException>();
    }

    [Test]
    public async Task EqualGroupsRemainExactAndFiniteExtremesDoNotOverflow()
    {
        var codec = BlockMeanCodec.Create();
        float[] weights = [float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue, -float.MaxValue];
        var decoded = codec.RoundTrip(weights, 1, weights.Length);

        await Assert.That(decoded).IsEquivalentTo(weights);
    }

    [Test]
    public async Task DirectLinearMatchesExpandedFp64AcrossGroupsAndShapes()
    {
        var random = new Random(611);
        foreach (var group in new[] { 2, 4, 7, 32, 1024 })
        {
            foreach (var columns in new[] { 1, 5, 137, 1025 })
            {
                const int Rows = 5;
                var codec = BlockMeanCodec.Create(group);
                var weights = Enumerable.Range(0, Rows * columns).Select(_ => (float)((random.NextDouble() * 4) - 2)).ToArray();
                var input = Enumerable.Range(0, columns).Select(_ => (float)((random.NextDouble() * 4) - 2)).ToArray();
                var encoded = new byte[codec.GetEncodedLength(Rows, columns)];
                codec.Encode(weights, Rows, columns, encoded);
                var decoded = codec.RoundTrip(weights, Rows, columns);
                var output = new float[Rows];

                codec.Multiply(encoded, Rows, columns, input, output);

                for (var row = 0; row < Rows; row++)
                {
                    var expected = Enumerable.Range(0, columns).Sum(column => (double)decoded[(row * columns) + column] * input[column]);
                    await Assert.That(Math.Abs(output[row] - expected)).IsLessThanOrEqualTo(2e-6 * Math.Max(1, Math.Abs(expected)));
                }
            }
        }
    }

    [Test]
    public async Task NonFiniteSourcesLeaveEncodedDestinationUnchanged()
    {
        var codec = BlockMeanCodec.Create();
        var encoded = Enumerable.Repeat((byte)0xAB, 8).ToArray();
        foreach (var invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            float[] weights = [1, 2, 3, 4, invalid];
            await Assert.That(() => codec.Encode(weights, 1, 5, encoded)).Throws<QuantizationException>();
        }

        await Assert.That(encoded.All(value => value == 0xAB)).IsTrue();
    }

    [Test]
    public async Task InvalidMeansAndInputsLeaveLinearAndDecodeOutputUnchanged()
    {
        var codec = BlockMeanCodec.Create();
        var encoded = new byte[8];
        var decoded = Enumerable.Repeat(42f, 8).ToArray();
        var output = new[] { 43f };
        foreach (var invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            BinaryPrimitives.WriteSingleLittleEndian(encoded.AsSpan()[4..], invalid);
            await Assert.That(() => codec.Decode(encoded, 1, 8, decoded)).Throws<FormatException>();
            await Assert.That(() => codec.Multiply(encoded, 1, 8, new float[8], output)).Throws<FormatException>();
        }

        encoded.AsSpan().Clear();
        var input = new float[8];
        input[^1] = float.NaN;
        await Assert.That(() => codec.Multiply(encoded, 1, 8, input, output)).Throws<ArgumentException>();
        await Assert.That(decoded.All(value => value == 42f)).IsTrue();
        await Assert.That(output[0]).IsEqualTo(43f);
    }

    [Test]
    public async Task Fp32LinearOverflowLeavesAllRowsUnchanged()
    {
        var codec = BlockMeanCodec.Create();
        float[] weights = [1, 1, 1, 1, float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue];
        var encoded = new byte[8];
        var output = new[] { 41f, 42f };
        codec.Encode(weights, 2, 4, encoded);

        await Assert.That(() => codec.Multiply(encoded, 2, 4, [2, 2, 2, 2], output)).Throws<ArithmeticException>();
        float[] expected = [41, 42];
        await Assert.That(output).IsEquivalentTo(expected);
    }

    [Test]
    public async Task EncoderAndDecoderRejectOverlappingStorage()
    {
        var codec = BlockMeanCodec.Create();
        var storage = new float[] { 1, 2, 3, 4 };
        var original = storage.ToArray();

        await Assert.That(() => codec.Encode(storage, 1, 4, MemoryMarshal.AsBytes(storage.AsSpan())[..4])).Throws<ArgumentException>();
        await Assert.That(() => codec.Decode(MemoryMarshal.AsBytes(storage.AsSpan())[..4], 1, 4, storage)).Throws<ArgumentException>();
        await Assert.That(storage).IsEquivalentTo(original);
    }

    [Test]
    public async Task LinearSupportsInputOutputAndEncodedOutputAliasing()
    {
        var codec = BlockMeanCodec.Create();
        float[] weights = [1, 1, 1, 1, 2, 2, 2, 2];
        var encoded = new byte[8];
        codec.Encode(weights, 2, 4, encoded);
        float[] input = [1, 2, 3, 4];

        codec.Multiply(encoded, 2, 4, input, input.AsSpan(0, 2));
        float[] expected = [10, 20];
        await Assert.That(input.Take(2)).IsEquivalentTo(expected);
        codec.Multiply(encoded, 2, 4, [1, 2, 3, 4], MemoryMarshal.Cast<byte, float>(encoded.AsSpan()));
        await Assert.That(MemoryMarshal.Cast<byte, float>(encoded).ToArray()).IsEquivalentTo(expected);
    }

    [Test]
    public async Task CallerScratchMatchesDefaultAndRejectsShortOrAliasedBuffers()
    {
        var codec = BlockMeanCodec.Create();
        var encoded = new byte[8];
        codec.Encode([1, 2, 3, 4, 5], 1, 5, encoded);
        float[] input = [1, 2, 3, 4, 5];
        var output = new[] { 42f };
        var sums = new double[2];
        var scratch = new float[1];

        codec.Multiply(encoded, 1, 5, input, output, sums, scratch);
        await Assert.That(output[0]).IsEqualTo(50f);
        await Assert.That(() => codec.Multiply(encoded, 1, 5, input, output, new double[1], scratch)).Throws<ArgumentException>();
        await Assert.That(() => codec.Multiply(encoded, 1, 5, input, output, sums, output)).Throws<ArgumentException>();
        await Assert.That(() => codec.Multiply(encoded, 1, 5, input, output, sums, input.AsSpan(0, 1))).Throws<ArgumentException>();
    }

    [Test]
    public async Task RepeatedCallerScratchLinearHasNoManagedAllocations()
    {
        var codec = BlockMeanCodec.Create();
        var encoded = new byte[codec.GetEncodedLength(8, 32)];
        var weights = Enumerable.Range(0, 8 * 32).Select(index => index * 0.01f).ToArray();
        var input = Enumerable.Repeat(0.5f, 32).ToArray();
        var output = new float[8];
        var sums = new double[8];
        var scratch = new float[8];
        codec.Encode(weights, 8, 32, encoded);
        codec.Multiply(encoded, 8, 32, input, output, sums, scratch);
        var before = GC.GetAllocatedBytesForCurrentThread();

        for (var iteration = 0; iteration < 100; iteration++)
        {
            codec.Multiply(encoded, 8, 32, input, output, sums, scratch);
        }

        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        await Assert.That(allocated).IsEqualTo(0L);
    }

    [Test]
    public async Task AveragingIsLossyAndOutputDistortionIsVisible()
    {
        var codec = BlockMeanCodec.Create();
        float[] weights = [1, -1, 1, -1];
        float[] probe = [1, -1, 1, -1];

        var distortion = WeightSensitivity.MeasureCodec(codec, weights, 1, 4, probe);

        await Assert.That(distortion.Relative).IsGreaterThan(0.99);
    }
}
