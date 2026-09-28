namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Rotary-position scaling methods that change the model math at every position (ADR-013).</summary>
public enum RopeScalingKind
{
    /// <summary>YaRN: interpolated low frequencies, extrapolated high frequencies, and a magnitude correction.</summary>
    Yarn,
}

/// <summary>An explicit rotary scaling profile that extends a model beyond its trained context (ADR-013).</summary>
/// <param name="Kind">Scaling method.</param>
/// <param name="Factor">Context extension factor; greater than one.</param>
/// <param name="OriginalContextLength">Context length the model was trained with.</param>
public sealed record RopeScaling(RopeScalingKind Kind, float Factor, int OriginalContextLength)
{
    /// <summary>Largest position count this profile covers: <c>floor(Factor × OriginalContextLength)</c>.</summary>
    public int ExtendedContextLength => checked((int)Math.Floor((double)Factor * OriginalContextLength));

    /// <summary>Stable profile suffix, for example <c>yarn4</c>.</summary>
    public string Name => "yarn" + Factor.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>Creates a validated YaRN profile.</summary>
    public static RopeScaling Yarn(float factor, int originalContextLength)
    {
        if (!float.IsFinite(factor) || factor <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(factor), factor, "A YaRN factor must be finite and greater than one.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(originalContextLength);
        var scaling = new RopeScaling(RopeScalingKind.Yarn, factor, originalContextLength);
        _ = scaling.ExtendedContextLength;
        return scaling;
    }
}
