namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Deterministic scalar matrix operations for the portable reference path.</summary>
public static class ReferenceLinearOperators
{
    /// <summary>Computes a row-major FP32 matrix-vector product with optional bias.</summary>
    /// <remarks>Uses a fixed left-to-right FP64 accumulation order and rounds each output once to FP32.</remarks>
    public static void Multiply(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias,
        Span<float> output)
    {
        if (input.IsEmpty || output.IsEmpty ||
            weights.Length != checked(input.Length * output.Length) ||
            (!bias.IsEmpty && bias.Length != output.Length))
        {
            throw new ArgumentException("Linear expects input [in], weights [out,in], and optional bias [out].");
        }

        if (!AllFinite(input) || !AllFinite(weights) || !AllFinite(bias))
        {
            throw new ReferenceNumericalException(
                ReferenceNumericalFailure.NonFiniteInput,
                "Linear has a non-finite active operand.");
        }

        var result = new float[output.Length];
        for (var row = 0; row < output.Length; row++)
        {
            var sum = bias.IsEmpty ? 0.0 : bias[row];
            for (var column = 0; column < input.Length; column++)
            {
                sum += (double)input[column] * weights[(row * input.Length) + column];
            }

            if (Math.Abs(sum) > float.MaxValue)
            {
                throw new ReferenceNumericalException(
                    ReferenceNumericalFailure.Fp32Overflow,
                    "Linear output exceeds the finite FP32 range.");
            }

            result[row] = (float)sum;
        }

        result.CopyTo(output);
    }

    private static bool AllFinite(ReadOnlySpan<float> values)
    {
        foreach (var value in values)
        {
            if (!float.IsFinite(value))
            {
                return false;
            }
        }

        return true;
    }
}
