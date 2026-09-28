namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Scalar normalization operations for the portable reference path.</summary>
public static class ReferenceNormalizationOperators
{
    /// <summary>Applies weighted RMS normalization with explicit epsilon.</summary>
    public static void RmsNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights,
        float epsilon,
        Span<float> output)
    {
        if (input.IsEmpty || input.Length != weights.Length || output.Length != input.Length ||
            !float.IsFinite(epsilon) || epsilon <= 0)
        {
            throw new ArgumentException("RMSNorm requires matching nonempty buffers and positive finite epsilon.");
        }

        var squared = 0.0;
        for (var index = 0; index < input.Length; index++)
        {
            if (!float.IsFinite(input[index]) || !float.IsFinite(weights[index]))
            {
                throw new ReferenceNumericalException(
                    ReferenceNumericalFailure.NonFiniteInput,
                    "RMSNorm has a non-finite active operand.");
            }

            squared += (double)input[index] * input[index];
        }

        var scale = 1.0 / Math.Sqrt((squared / input.Length) + epsilon);
        var result = new float[output.Length];
        for (var index = 0; index < result.Length; index++)
        {
            var value = input[index] * scale * weights[index];
            if (!double.IsFinite(value) || Math.Abs(value) > float.MaxValue)
            {
                throw new ReferenceNumericalException(
                    ReferenceNumericalFailure.Fp32Overflow,
                    "RMSNorm output exceeds the finite FP32 range.");
            }

            result[index] = (float)value;
        }

        result.CopyTo(output);
    }
}
