using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class ReferenceOperatorsTests
{
    [Test]
    public async Task OperatorsMatchFp64Oracle()
    {
        var random = new Random(18077);
        for (var sample = 0; sample < 32; sample++)
        {
            var inputWidth = 1 + (sample % 8);
            var outputWidth = 1 + (sample % 5);
            var input = Enumerable.Range(0, inputWidth)
                .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
            var weights = Enumerable.Range(0, inputWidth * outputWidth)
                .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
            var bias = Enumerable.Range(0, outputWidth)
                .Select(_ => (float)((random.NextDouble() * 2) - 1)).ToArray();
            if (sample == 31)
            {
                input = [10_000f, 0.0001f, -10_000f, 1f, -1f, 2f, -2f, 0.5f];
            }

            var actual = new float[outputWidth];
            ReferenceLinearOperators.Multiply(input, weights, bias, actual);

            for (var row = 0; row < outputWidth; row++)
            {
                var expected = (double)bias[row];
                for (var column = 0; column < inputWidth; column++)
                {
                    expected += (double)input[column] * weights[(row * inputWidth) + column];
                }

                await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(actual[row], expected)).IsTrue();
            }
        }
    }

    [Test]
    public async Task LinearRejectsMalformedShapeBeforeWriting()
    {
        float[] output = [42f, 43f];

        await Assert.That(() => ReferenceLinearOperators.Multiply(
            [1f, 2f], [3f, 4f, 5f], [0f, 0f], output)).Throws<ArgumentException>();
        await Assert.That(output).IsEquivalentTo([42f, 43f]);
    }

    [Test]
    public async Task CausalMaskPreventsFutureLeak()
    {
        float[] query = [1f, 0f, 0f, 1f];
        float[] keys = [1f, 0f, 0f, 1f, 0.5f, 0.5f];
        float[] values = [10f, 20f, 30f, 40f, 50f, 60f];
        bool[] valid = [true, true, true];
        var before = new float[4];
        ReferenceAttentionOperators.Execute(
            query, keys, values, valid, keyValueHeads: 1,
            headDimension: 2, queryPosition: 1, scale: 1f, before);

        keys[4] = float.NaN;
        keys[5] = 1000f;
        values[4] = float.NaN;
        values[5] = -1_000_000f;
        var after = new float[4];
        ReferenceAttentionOperators.Execute(
            query, keys, values, valid, keyValueHeads: 1,
            headDimension: 2, queryPosition: 1, scale: 1f, after);

        await Assert.That(after).IsEquivalentTo(before);
        var firstWeight = Math.Exp(1) / (Math.Exp(1) + 1);
        var expectedFirst = (firstWeight * 10) + ((1 - firstWeight) * 30);
        var expectedSecondHead = ((1 - firstWeight) * 10) + (firstWeight * 30);
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(before[0], expectedFirst)).IsTrue();
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(before[2], expectedSecondHead)).IsTrue();
    }

    [Test]
    public async Task AllMaskedRowDefined()
    {
        float[] output = [42f, 43f];
        ReferenceNumericalException? failure = null;

        try
        {
            ReferenceAttentionOperators.Execute(
                [1f, 0f], [1f, 0f, 0f, 1f], [10f, 20f, 30f, 40f],
                [false, false], keyValueHeads: 1, headDimension: 2,
                queryPosition: 1, scale: 1f, output);
        }
        catch (ReferenceNumericalException exception)
        {
            failure = exception;
        }

        await Assert.That(failure?.Failure == ReferenceNumericalFailure.AllKeysMasked).IsTrue();
        await Assert.That(output).IsEquivalentTo([42f, 43f]);
    }

    [Test]
    public async Task RmsNormAndSiluMatchFp64Oracle()
    {
        float[] input = [0.25f, -2f, 3.5f, 0f];
        float[] weights = [1f, 0.5f, -1.25f, 2f];
        var normalized = new float[input.Length];
        var activated = new float[input.Length];

        ReferenceNormalizationOperators.RmsNorm(input, weights, 1e-5f, normalized);
        ReferenceVectorOperators.Silu(input, activated);

        var sumOfSquares = input.Sum(value => (double)value * value);
        var scale = 1.0 / Math.Sqrt((sumOfSquares / input.Length) + 1e-5f);
        for (var index = 0; index < input.Length; index++)
        {
            var normOracle = input[index] * scale * weights[index];
            var siluOracle = input[index] / (1 + Math.Exp(-input[index]));
            await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(normalized[index], normOracle)).IsTrue();
            await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(activated[index], siluOracle)).IsTrue();
        }
    }

    [Test]
    public async Task StableSoftmaxAvoidsOverflow()
    {
        float[] logits = [1000f, 1001f, -1000f];
        var output = new float[logits.Length];

        ReferenceVectorOperators.Softmax(logits, output);

        var first = 1.0 / (1 + Math.E);
        var second = Math.E / (1 + Math.E);
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(output[0], first)).IsTrue();
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(output[1], second)).IsTrue();
        await Assert.That(output[2]).IsEqualTo(0f);
        await Assert.That(output.Sum()).IsBetween(0.99999f, 1.00001f);
    }

    [Test]
    public async Task RopeLayoutsMatchKnownCoordinates()
    {
        float[] input = [1f, 2f, 3f, 4f];
        var neox = new float[4];
        var interleaved = new float[4];

        ReferenceRotaryOperators.Apply(input, 4, 1, 10_000f, RotaryLayout.NeoX, neox);
        ReferenceRotaryOperators.Apply(input, 4, 1, 10_000f, RotaryLayout.Interleaved, interleaved);

        var cosine = Math.Cos(1);
        var sine = Math.Sin(1);
        var neoxFirst = cosine - (3 * sine);
        var interleavedFirst = cosine - (2 * sine);
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(neox[0], neoxFirst)).IsTrue();
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(interleaved[0], interleavedFirst)).IsTrue();
        var inputNorm = input.Sum(value => (double)value * value);
        var neoxNorm = neox.Sum(value => (double)value * value);
        var interleavedNorm = interleaved.Sum(value => (double)value * value);
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance((float)neoxNorm, inputNorm)).IsTrue();
        await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance((float)interleavedNorm, inputNorm)).IsTrue();
    }

    [Test]
    public async Task ElementwiseRejectsShapeMismatchBeforeWriting()
    {
        float[] output = [42f, 43f];

        await Assert.That(() => ReferenceVectorOperators.Add(
            [1f, 2f], [3f], output)).Throws<ArgumentException>();
        await Assert.That(output).IsEquivalentTo([42f, 43f]);
    }
}
