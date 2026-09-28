using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

public sealed class ActivationQuantizerTests
{
    [Test]
    public async Task ActivationQuantizerMatchesGgmlContract()
    {
        var random = new Random(20260928);
        foreach (var columns in new[] { 32, 896, 4864 })
        {
            var input = Enumerable.Range(0, columns)
                .Select(_ => (float)((random.NextDouble() * 8) - 4) * (float)random.NextDouble())
                .ToArray();
            var quants = new sbyte[columns];
            var scales = new float[columns / 32];

            Q8ActivationQuantizer.QuantizeRow(input, quants, scales);

            var (expectedQuants, expectedScales) = Expected(input);
            await Assert.That(quants).IsEquivalentTo(expectedQuants);
            await Assert.That(scales).IsEquivalentTo(expectedScales);
        }
    }

    [Test]
    public async Task ActivationQuantizerRoundsTiesToEven()
    {
        var input = new float[32];
        input[0] = 127f;
        input[1] = 2.5f;
        input[2] = 3.5f;
        input[3] = -2.5f;
        input[4] = -0.5f;
        input[5] = 0.5f;
        var quants = new sbyte[32];
        var scales = new float[1];

        Q8ActivationQuantizer.QuantizeRow(input, quants, scales);

        await Assert.That(scales[0]).IsEqualTo(1f);
        await Assert.That(quants[..6]).IsEquivalentTo(new sbyte[] { 127, 2, 4, -2, 0, 0 });
    }

    [Test]
    public async Task ZeroBlockStoresZeroScaleAndCodes()
    {
        var input = new float[64];
        input[40] = -3f;
        var quants = new sbyte[64];
        quants.AsSpan().Fill(9);
        var scales = new float[] { 7f, 7f };

        Q8ActivationQuantizer.QuantizeRow(input, quants, scales);

        await Assert.That(scales[0]).IsEqualTo(0f);
        await Assert.That(quants[..32].All(value => value == 0)).IsTrue();
        await Assert.That(quants[40]).IsEqualTo((sbyte)-127);
        await Assert.That(scales[1]).IsEqualTo((float)(Half)(3f / 127f));
    }

    [Test]
    public async Task QuantizerRejectsMisalignedShapes()
    {
        await Assert.That(() => Q8ActivationQuantizer.QuantizeRow(new float[33], new sbyte[33], new float[1]))
            .Throws<ArgumentException>();
        await Assert.That(() => Q8ActivationQuantizer.QuantizeRow(new float[64], new sbyte[64], new float[1]))
            .Throws<ArgumentException>();
    }

    private static (sbyte[] Quants, float[] Scales) Expected(float[] input)
    {
        var quants = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        for (var block = 0; block < scales.Length; block++)
        {
            var values = input.AsSpan(block * 32, 32);
            var maximum = 0f;
            foreach (var value in values)
            {
                maximum = MathF.Max(maximum, MathF.Abs(value));
            }

            var scale = maximum / 127f;
            var inverse = scale == 0 ? 0 : 1f / scale;
            scales[block] = (float)(Half)scale;
            for (var index = 0; index < 32; index++)
            {
                quants[(block * 32) + index] = (sbyte)MathF.Round(values[index] * inverse, MidpointRounding.ToEven);
            }
        }

        return (quants, scales);
    }
}
