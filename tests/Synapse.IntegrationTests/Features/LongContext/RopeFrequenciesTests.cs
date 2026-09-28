using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

public sealed class RopeFrequenciesTests
{
    private const int HeadDimension = 64;
    private const float Theta = 1_000_000f;
    private static readonly int[] Positions = [0, 1, 17, 4_095, 32_767, 65_536, 131_071];

    [Test]
    public async Task UnscaledFrequenciesStayBitwise()
    {
        var frequencies = new RopeFrequencies(HeadDimension, Theta, scaling: null);
        var cosines = new float[HeadDimension / 2];
        var sines = new float[HeadDimension / 2];
        var mismatches = 0;

        foreach (var position in Positions)
        {
            frequencies.Compute(position, cosines, sines);
            for (var index = 0; index < cosines.Length; index++)
            {
                var angle = position / MathF.Pow(Theta, 2.0f * index / HeadDimension);
                mismatches += BitConverter.SingleToInt32Bits(cosines[index]) == BitConverter.SingleToInt32Bits(MathF.Cos(angle)) &&
                    BitConverter.SingleToInt32Bits(sines[index]) == BitConverter.SingleToInt32Bits(MathF.Sin(angle))
                    ? 0
                    : 1;
            }
        }

        await Assert.That(mismatches).IsEqualTo(0);
    }

    [Test]
    public async Task YarnFrequenciesMatchFp64Formula()
    {
        const float Factor = 4;
        const int Original = 32_768;
        var frequencies = new RopeFrequencies(HeadDimension, Theta, RopeScaling.Yarn(Factor, Original));
        var cosines = new float[HeadDimension / 2];
        var sines = new float[HeadDimension / 2];
        var worst = 0.0;

        foreach (var position in Positions)
        {
            frequencies.Compute(position, cosines, sines);
            for (var index = 0; index < cosines.Length; index++)
            {
                var (cosine, sine, angle) = Oracle(position, index, Factor, Original);
                var tolerance = (8 * Math.Abs(angle) * Math.Pow(2, -23)) + 2e-6;
                worst = Math.Max(worst, Math.Abs(cosines[index] - cosine) / tolerance);
                worst = Math.Max(worst, Math.Abs(sines[index] - sine) / tolerance);
            }
        }

        await Assert.That(worst).IsLessThanOrEqualTo(1.0);
    }

    /// <summary>FP64 evaluation of ggml <c>rope_yarn</c> with beta_fast 32, beta_slow 1, and ext_factor 1.</summary>
    private static (double Cosine, double Sine, double Angle) Oracle(int position, int index, double factor, int original)
    {
        var extrapolated = position / Math.Pow(Theta, 2.0 * index / HeadDimension);
        var interpolated = extrapolated / factor;
        var low = Math.Max(0, Math.Floor(CorrectionDimension(original, 32)));
        var high = Math.Min(HeadDimension - 1, Math.Ceiling(CorrectionDimension(original, 1)));
        var ramp = 1 - Math.Clamp((index - low) / Math.Max(0.001, high - low), 0, 1);
        var angle = (interpolated * (1 - ramp)) + (extrapolated * ramp);
        var magnitude = 1 + (0.1 * Math.Log(factor));
        return (Math.Cos(angle) * magnitude, Math.Sin(angle) * magnitude, angle);
    }

    private static double CorrectionDimension(int original, double rotations) =>
        HeadDimension * Math.Log(original / (rotations * 2 * Math.PI)) / (2 * Math.Log(Theta));
}
