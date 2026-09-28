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
}
