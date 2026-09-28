namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>Activation-weighted output distortion of one approximated matrix.</summary>
/// <param name="ErrorEnergy">Sum over samples of <c>||(W - A) x||²</c>.</param>
/// <param name="SignalEnergy">Sum over samples of <c>||W x||²</c>.</param>
/// <param name="Samples">Number of calibration vectors.</param>
public readonly record struct OutputDistortion(double ErrorEnergy, double SignalEnergy, int Samples)
{
    /// <summary>Stabilizer added to the signal energy.</summary>
    public const double Epsilon = 1e-12;

    /// <summary>Relative distortion <c>E = error / (signal + epsilon)</c>.</summary>
    public double Relative => ErrorEnergy / (SignalEnergy + Epsilon);
}

/// <summary>
/// Heuristic weight importance from the specification (§6.2): the output error an
/// approximation causes on real calibration activations, not weight magnitude alone.
/// It ranks candidates; held-out ablation and task quality remain the qualification.
/// </summary>
public static class WeightSensitivity
{
    /// <summary>Measures distortion of <paramref name="approximation"/> against <paramref name="reference"/>.</summary>
    public static OutputDistortion Measure(
        ReadOnlySpan<float> reference,
        ReadOnlySpan<float> approximation,
        int rows,
        int columns,
        ReadOnlySpan<float> calibration)
    {
        ValidateShapes(reference, approximation, rows, columns, calibration);
        var samples = calibration.Length / columns;
        var error = 0.0;
        var signal = 0.0;
        for (var sample = 0; sample < samples; sample++)
        {
            var input = calibration.Slice(sample * columns, columns);
            for (var row = 0; row < rows; row++)
            {
                var referenceRow = reference.Slice(row * columns, columns);
                var approximationRow = approximation.Slice(row * columns, columns);
                var output = 0.0;
                var difference = 0.0;
                for (var column = 0; column < columns; column++)
                {
                    output += (double)referenceRow[column] * input[column];
                    difference += ((double)referenceRow[column] - approximationRow[column]) * input[column];
                }

                signal += output * output;
                error += difference * difference;
            }
        }

        return new OutputDistortion(error, signal, samples);
    }

    /// <summary>Quantizes on the fly with <paramref name="codec"/> and measures the result.</summary>
    public static OutputDistortion MeasureCodec(
        IWeightCodec codec,
        ReadOnlySpan<float> weights,
        int rows,
        int columns,
        ReadOnlySpan<float> calibration) =>
        Measure(weights, codec.RoundTrip(weights, rows, columns), rows, columns, calibration);

    private static void ValidateShapes(
        ReadOnlySpan<float> reference,
        ReadOnlySpan<float> approximation,
        int rows,
        int columns,
        ReadOnlySpan<float> calibration)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        var length = checked(rows * columns);
        if (reference.Length != length || approximation.Length != length)
        {
            throw new ArgumentException($"Sensitivity expects two [{rows},{columns}] matrices.");
        }

        if (calibration.IsEmpty || calibration.Length % columns != 0)
        {
            throw new ArgumentException($"Calibration must contain one or more vectors of length {columns}.");
        }

        if (!AllFinite(reference) || !AllFinite(approximation) || !AllFinite(calibration))
        {
            throw new ArgumentException("Sensitivity inputs must be finite.");
        }
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
