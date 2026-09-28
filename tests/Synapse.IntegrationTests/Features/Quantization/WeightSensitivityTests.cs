using ManagedCode.Synapse.Runtime.Features.Quantization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class WeightSensitivityTests
{
    [Test]
    public async Task DistortionMatchesFp64Formula()
    {
        float[] reference = [1f, 2f, 3f, 4f];
        float[] approximation = [1.5f, 2f, 3f, 3f];
        float[] calibration = [1f, 1f, 0.5f, -1f];

        var measured = WeightSensitivity.Measure(reference, approximation, 2, 2, calibration);

        // Samples: x0=[1,1], x1=[0.5,-1]. Error rows are [0.5,0] and [0,-1].
        var error = Square(0.5) + Square(-1) + Square(0.25) + Square(1);
        var signal = Square(3) + Square(7) + Square(-1.5) + Square(-2.5);
        await Assert.That(measured.Samples).IsEqualTo(2);
        await Assert.That(Math.Abs(measured.ErrorEnergy - error)).IsLessThan(1e-12);
        await Assert.That(Math.Abs(measured.SignalEnergy - signal)).IsLessThan(1e-12);
        await Assert.That(Math.Abs(measured.Relative - (error / signal))).IsLessThan(1e-12);
    }

    [Test]
    public async Task DistortionIsActivationAware()
    {
        float[] reference = [0.3f, 1f, 0.3f, 1f];
        var approximation = WeightCodecs.Ternary.RoundTrip(reference, 2, 2);
        float[] quietColumn = [0f, 1f, 0f, 2f];
        float[] activeColumn = [1f, 0f, 2f, 0f];

        var whenQuiet = WeightSensitivity.Measure(reference, approximation, 2, 2, quietColumn);
        var whenActive = WeightSensitivity.Measure(reference, approximation, 2, 2, activeColumn);

        await Assert.That(whenQuiet.Relative).IsLessThan(whenActive.Relative);
    }

    [Test]
    public async Task ExactlyRepresentableWeightsHaveZeroDistortion()
    {
        var weights = Enumerable.Range(0, 64).Select(index => (float)((index % 15) - 7)).ToArray();
        var calibration = Enumerable.Range(0, 128).Select(index => (float)Math.Sin(index)).ToArray();

        var measured = WeightSensitivity.MeasureCodec(WeightCodecs.Q4, weights, 1, 64, calibration);

        await Assert.That(measured.ErrorEnergy).IsEqualTo(0.0);
        await Assert.That(measured.Relative).IsEqualTo(0.0);
    }

    [Test]
    public async Task LowerPrecisionCodecHasHigherDistortion()
    {
        var random = new Random(77);
        var weights = Enumerable.Range(0, 8 * 128)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
        var calibration = Enumerable.Range(0, 16 * 128)
            .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();

        var q4 = WeightSensitivity.MeasureCodec(WeightCodecs.Q4, weights, 8, 128, calibration);
        var ternary = WeightSensitivity.MeasureCodec(WeightCodecs.Ternary, weights, 8, 128, calibration);

        await Assert.That(q4.Relative).IsGreaterThan(0.0);
        await Assert.That(ternary.Relative).IsGreaterThan(q4.Relative);
    }

    [Test]
    public async Task CalibrationShapeValidated()
    {
        float[] weights = [1f, 2f];

        await Assert.That(() => WeightSensitivity.Measure(weights, weights, 1, 2, [])).Throws<ArgumentException>();
        await Assert.That(() => WeightSensitivity.Measure(weights, weights, 1, 2, [1f, 2f, 3f])).Throws<ArgumentException>();
        await Assert.That(() => WeightSensitivity.Measure(weights, weights, 1, 2, [float.NaN, 1f]))
            .Throws<ArgumentException>();
    }

    private static double Square(double value) => value * value;
}
