using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class PrecisionCandidatesTests
{
    [Test]
    public async Task MeasuredOptionsCoverAnyBitWidthInSizeOrder()
    {
        var (weights, calibration) = Data();
        var source = new PrecisionOption("fp32", 4 * 128 * sizeof(float), 0);
        IWeightCodec[] codecs =
        [
            SymmetricGroupCodec.Create(2, 64),
            WeightCodecs.Q4,
            SymmetricGroupCodec.Create(8, 64),
            SymmetricGroupCodec.Create(6, 64),
            SymmetricGroupCodec.Create(3, 64),
        ];

        var candidate = PrecisionCandidates.Measure(new TensorId(9), source, weights, 4, 128, calibration, codecs);

        var bytes = candidate.Options.Select(option => option.Bytes).ToArray();
        await Assert.That(candidate.Options[0]).IsEqualTo(source);
        await Assert.That(bytes).IsEquivalentTo(new long[] { 2048, 528, 400, 272, 208, 144 });
        await Assert.That(candidate.Options.Skip(1).All(option => option.Distortion > 0)).IsTrue();
        await Assert.That(candidate.Options.Zip(candidate.Options.Skip(1)).All(pair => pair.First.Distortion <= pair.Second.Distortion)).IsTrue();
    }

    [Test]
    public async Task EqualSizeOptionsKeepLowerDistortion()
    {
        var (weights, calibration) = Data();
        var source = new PrecisionOption("fp32", 4 * 128 * sizeof(float), 0);
        var twoBit = SymmetricGroupCodec.Create(2, 64);

        var candidate = PrecisionCandidates.Measure(
            new TensorId(1), source, weights, 4, 128, calibration, [twoBit, WeightCodecs.Ternary]);

        var twoBitDistortion = WeightSensitivity.MeasureCodec(twoBit, weights, 4, 128, calibration).Relative;
        var ternaryDistortion = WeightSensitivity.MeasureCodec(WeightCodecs.Ternary, weights, 4, 128, calibration).Relative;
        var kept = candidate.Options.Single(option => option.Bytes == 144);
        await Assert.That(candidate.Options.Count).IsEqualTo(2);
        await Assert.That(kept.Distortion).IsEqualTo(Math.Min(twoBitDistortion, ternaryDistortion));
    }

    [Test]
    public async Task EncodingsNotSmallerThanSourceAreDropped()
    {
        var (weights, calibration) = Data();
        var source = new PrecisionOption("gguf.q4_0", 300, 0);

        var candidate = PrecisionCandidates.Measure(
            new TensorId(2), source, weights, 4, 128, calibration, [SymmetricGroupCodec.Create(8, 64), WeightCodecs.Q4]);

        await Assert.That(candidate.Options.Select(option => option.EncodingId))
            .IsEquivalentTo(["gguf.q4_0", SynQ4BlockCodec.EncodingId]);
    }

    private static (float[] Weights, float[] Calibration) Data()
    {
        var random = new Random(404);
        var weights = Enumerable.Range(0, 4 * 128).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var calibration = Enumerable.Range(0, 6 * 128).Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        return (weights, calibration);
    }
}
