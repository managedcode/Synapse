using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>Builds a tensor's precision options by quantizing it on the fly with each codec.</summary>
public static class PrecisionCandidates
{
    /// <summary>
    /// Measures every codec against the source weights and returns the options in strictly
    /// decreasing stored size. Encodings not smaller than the source are dropped; among
    /// encodings of equal size, the one with lower distortion is kept.
    /// </summary>
    public static PrecisionCandidate Measure(
        TensorId tensor,
        PrecisionOption source,
        ReadOnlySpan<float> weights,
        int rows,
        int columns,
        ReadOnlySpan<float> calibration,
        IEnumerable<IWeightCodec> codecs,
        bool pinnedHigh = false)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(codecs);
        var measured = new List<PrecisionOption>();
        foreach (var codec in codecs)
        {
            ArgumentNullException.ThrowIfNull(codec);
            var bytes = codec.GetEncodedLength(rows, columns);
            if (bytes >= source.Bytes)
            {
                continue;
            }

            var distortion = WeightSensitivity.MeasureCodec(codec, weights, rows, columns, calibration).Relative;
            measured.Add(new PrecisionOption(codec.EncodingId, bytes, distortion));
        }

        var options = new List<PrecisionOption> { source };
        foreach (var option in measured
                     .OrderByDescending(option => option.Bytes)
                     .ThenBy(option => option.Distortion)
                     .ThenBy(option => option.EncodingId, StringComparer.Ordinal))
        {
            if (option.Bytes < options[^1].Bytes)
            {
                options.Add(option);
            }
        }

        return new PrecisionCandidate(tensor, options.AsReadOnly(), pinnedHigh);
    }
}
