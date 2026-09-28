using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Scalar rotary coordinate transforms for the portable reference path.</summary>
public static class ReferenceRotaryOperators
{
    /// <summary>Applies NeoX or interleaved RoPE to flattened attention heads.</summary>
    public static void Apply(
        ReadOnlySpan<float> input,
        int headDimension,
        int position,
        float theta,
        RotaryLayout layout,
        Span<float> output)
    {
        if (input.IsEmpty || headDimension <= 0 || (headDimension & 1) != 0 ||
            input.Length % headDimension != 0 || output.Length != input.Length ||
            position < 0 || !float.IsFinite(theta) || theta <= 0 || !Enum.IsDefined(layout))
        {
            throw new ArgumentException("RoPE requires even head width, nonnegative position, positive theta, and matching buffers.");
        }

        foreach (var value in input)
        {
            if (!float.IsFinite(value))
            {
                throw new ReferenceNumericalException(
                    ReferenceNumericalFailure.NonFiniteInput,
                    "RoPE has a non-finite input coordinate.");
            }
        }

        var result = new float[input.Length];
        var half = headDimension / 2;
        for (var head = 0; head < input.Length / headDimension; head++)
        {
            for (var pair = 0; pair < half; pair++)
            {
                var first = (head * headDimension) + (layout == RotaryLayout.NeoX ? pair : pair * 2);
                var second = first + (layout == RotaryLayout.NeoX ? half : 1);
                var angle = position / Math.Pow(theta, 2.0 * pair / headDimension);
                var cosine = Math.Cos(angle);
                var sine = Math.Sin(angle);
                result[first] = RoundFinite((input[first] * cosine) - (input[second] * sine));
                result[second] = RoundFinite((input[first] * sine) + (input[second] * cosine));
            }
        }

        result.CopyTo(output);
    }

    private static float RoundFinite(double value)
    {
        if (!double.IsFinite(value) || Math.Abs(value) > float.MaxValue)
        {
            throw new ReferenceNumericalException(
                ReferenceNumericalFailure.Fp32Overflow,
                "RoPE output exceeds the finite FP32 range.");
        }

        return (float)value;
    }
}
