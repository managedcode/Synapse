using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// The single source of rotary cosines and sines for every backend (ADR-013). Unscaled angles use the FP32
/// expression <c>position / MathF.Pow(theta, 2i/d)</c> bitwise. YaRN follows ggml <c>rope_yarn</c> with
/// beta_fast 32, beta_slow 1, ext_factor 1, and the magnitude correction <c>1 + 0.1 ln(factor)</c>.
/// </summary>
internal sealed class RopeFrequencies
{
    private const float BetaFast = 32;
    private const float BetaSlow = 1;
    private readonly int _headDimension;
    private readonly float _theta;
    private readonly float _factor;
    private readonly float _magnitude;
    private readonly float[]? _extrapolationMix;

    public RopeFrequencies(int headDimension, float theta, RopeScaling? scaling)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);
        if ((headDimension & 1) != 0 || !float.IsFinite(theta) || theta <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(headDimension), "RoPE needs an even head dimension and theta above one.");
        }

        _headDimension = headDimension;
        _theta = theta;
        Half = headDimension / 2;
        Scaling = scaling;
        if (scaling is null)
        {
            return;
        }

        _factor = scaling.Factor;
        _magnitude = 1.0f + (0.1f * MathF.Log(scaling.Factor));
        var low = MathF.Max(0, MathF.Floor(CorrectionDimension(scaling.OriginalContextLength, BetaFast)));
        var high = MathF.Min(headDimension - 1, MathF.Ceiling(CorrectionDimension(scaling.OriginalContextLength, BetaSlow)));
        _extrapolationMix = new float[Half];
        for (var index = 0; index < Half; index++)
        {
            var ramp = (index - low) / MathF.Max(0.001f, high - low);
            _extrapolationMix[index] = 1 - MathF.Min(1, MathF.Max(0, ramp));
        }
    }

    /// <summary>Rotated pairs per head.</summary>
    public int Half { get; }

    public RopeScaling? Scaling { get; }

    /// <summary>Writes the <see cref="Half"/> cosines and sines for <paramref name="position"/>.</summary>
    public void Compute(int position, Span<float> cosines, Span<float> sines)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (cosines.Length < Half || sines.Length < Half)
        {
            throw new ArgumentException($"Cosine and sine spans need {Half} values.", nameof(cosines));
        }

        for (var index = 0; index < Half; index++)
        {
            var angle = position / MathF.Pow(_theta, 2.0f * index / _headDimension);
            if (_extrapolationMix is null)
            {
                cosines[index] = MathF.Cos(angle);
                sines[index] = MathF.Sin(angle);
                continue;
            }

            var mix = _extrapolationMix[index];
            var scaled = (angle / _factor * (1 - mix)) + (angle * mix);
            cosines[index] = MathF.Cos(scaled) * _magnitude;
            sines[index] = MathF.Sin(scaled) * _magnitude;
        }
    }

    private float CorrectionDimension(int originalContext, float rotations) =>
        _headDimension * MathF.Log(originalContext / (rotations * 2 * MathF.PI)) / (2 * MathF.Log(_theta));
}
