namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Deterministic scalar element-wise and softmax operations.</summary>
public static class ReferenceVectorOperators
{
    /// <summary>Adds two equally sized finite vectors.</summary>
    public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) =>
        Binary(left, right, output, static (a, b) => a + b);

    /// <summary>Multiplies two equally sized finite vectors.</summary>
    public static void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output) =>
        Binary(left, right, output, static (a, b) => a * b);

    /// <summary>Applies the numerically stable SiLU activation.</summary>
    public static void Silu(ReadOnlySpan<float> input, Span<float> output)
    {
        ValidateUnary(input, output);
        var result = new float[output.Length];
        for (var index = 0; index < input.Length; index++)
        {
            var value = (double)input[index];
            var sigmoid = value >= 0
                ? 1.0 / (1.0 + Math.Exp(-value))
                : Math.Exp(value) / (1.0 + Math.Exp(value));
            result[index] = RoundFinite(value * sigmoid);
        }

        result.CopyTo(output);
    }

    /// <summary>Applies stable softmax to a finite nonempty vector.</summary>
    public static void Softmax(ReadOnlySpan<float> input, Span<float> output)
    {
        ValidateUnary(input, output);
        var maximum = input[0];
        for (var index = 1; index < input.Length; index++)
        {
            maximum = Math.Max(maximum, input[index]);
        }

        var result = new double[input.Length];
        var denominator = 0.0;
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = Math.Exp((double)input[index] - maximum);
            denominator += result[index];
        }

        var rounded = new float[output.Length];
        for (var index = 0; index < rounded.Length; index++)
        {
            rounded[index] = (float)(result[index] / denominator);
        }

        rounded.CopyTo(output);
    }

    private static void Binary(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> output,
        Func<double, double, double> operation)
    {
        if (left.IsEmpty || left.Length != right.Length || output.Length != left.Length)
        {
            throw new ArgumentException("Element-wise buffers must have the same nonzero length.");
        }

        var result = new float[output.Length];
        for (var index = 0; index < result.Length; index++)
        {
            if (!float.IsFinite(left[index]) || !float.IsFinite(right[index]))
            {
                throw NonFinite();
            }

            result[index] = RoundFinite(operation(left[index], right[index]));
        }

        result.CopyTo(output);
    }

    private static void ValidateUnary(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.IsEmpty || output.Length != input.Length)
        {
            throw new ArgumentException("Unary buffers must have the same nonzero length.");
        }

        foreach (var value in input)
        {
            if (!float.IsFinite(value))
            {
                throw NonFinite();
            }
        }
    }

    private static float RoundFinite(double value)
    {
        if (!double.IsFinite(value) || Math.Abs(value) > float.MaxValue)
        {
            throw new ReferenceNumericalException(
                ReferenceNumericalFailure.Fp32Overflow,
                "Vector output exceeds the finite FP32 range.");
        }

        return (float)value;
    }

    private static ReferenceNumericalException NonFinite() => new(
        ReferenceNumericalFailure.NonFiniteInput,
        "Vector operation has a non-finite active operand.");
}
