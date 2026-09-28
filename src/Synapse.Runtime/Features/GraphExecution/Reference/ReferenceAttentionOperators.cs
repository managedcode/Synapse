namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Scalar causal grouped-query attention over explicit KV positions.</summary>
public static class ReferenceAttentionOperators
{
    /// <summary>Computes one causal query step with a per-position validity mask.</summary>
    /// <remarks>Keys and values are row-major [position, kv-head, head-dimension].
    /// A masked or future position is never read. Accumulation uses FP64 and
    /// each output is rounded once to FP32.</remarks>
    public static void Execute(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values,
        ReadOnlySpan<bool> validPositions,
        int keyValueHeads,
        int headDimension,
        int queryPosition,
        float scale,
        Span<float> output)
    {
        var queryHeads = ValidateShape(
            query, keys, values, validPositions,
            keyValueHeads, headDimension, queryPosition, scale, output);
        ValidateActiveValues(query, keys, values, validPositions, queryPosition, keyValueHeads, headDimension);

        var result = new float[output.Length];
        var groupSize = queryHeads / keyValueHeads;
        for (var head = 0; head < queryHeads; head++)
        {
            ComputeHead(
                query.Slice(head * headDimension, headDimension),
                keys,
                values,
                validPositions,
                head / groupSize,
                keyValueHeads,
                headDimension,
                queryPosition,
                scale,
                result.AsSpan(head * headDimension, headDimension));
        }

        result.CopyTo(output);
    }

    private static int ValidateShape(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values,
        ReadOnlySpan<bool> valid,
        int kvHeads,
        int dimension,
        int position,
        float scale,
        Span<float> output)
    {
        if (kvHeads <= 0 || dimension <= 0 || query.IsEmpty || query.Length % dimension != 0 ||
            query.Length / dimension % kvHeads != 0 || output.Length != query.Length ||
            valid.IsEmpty || position < 0 || position >= valid.Length ||
            !float.IsFinite(scale) || scale <= 0 ||
            keys.Length != checked(valid.Length * kvHeads * dimension) || values.Length != keys.Length)
        {
            throw new ArgumentException("Causal attention buffers or parameters have incompatible shapes.");
        }

        return query.Length / dimension;
    }

    private static void ValidateActiveValues(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values,
        ReadOnlySpan<bool> valid,
        int position,
        int kvHeads,
        int dimension)
    {
        if (!AllFinite(query))
        {
            throw NonFinite();
        }

        var visible = false;
        var stride = kvHeads * dimension;
        for (var current = 0; current <= position; current++)
        {
            if (!valid[current])
            {
                continue;
            }

            visible = true;
            if (!AllFinite(keys.Slice(current * stride, stride)) ||
                !AllFinite(values.Slice(current * stride, stride)))
            {
                throw NonFinite();
            }
        }

        if (!visible)
        {
            throw new ReferenceNumericalException(
                ReferenceNumericalFailure.AllKeysMasked,
                "Causal attention has no valid key at this query position.");
        }
    }

    private static void ComputeHead(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values,
        ReadOnlySpan<bool> valid,
        int kvHead,
        int kvHeads,
        int dimension,
        int position,
        float scale,
        Span<float> output)
    {
        var scores = new double[position + 1];
        var maximum = double.NegativeInfinity;
        for (var current = 0; current <= position; current++)
        {
            if (!valid[current])
            {
                continue;
            }

            var key = keys.Slice(((current * kvHeads) + kvHead) * dimension, dimension);
            var score = 0.0;
            for (var index = 0; index < dimension; index++)
            {
                score += (double)query[index] * key[index];
            }

            scores[current] = score * scale;
            maximum = Math.Max(maximum, scores[current]);
        }

        NormalizeAndAccumulate(values, valid, scores, maximum, kvHead, kvHeads, dimension, position, output);
    }

    private static void NormalizeAndAccumulate(
        ReadOnlySpan<float> values,
        ReadOnlySpan<bool> valid,
        ReadOnlySpan<double> scores,
        double maximum,
        int kvHead,
        int kvHeads,
        int dimension,
        int position,
        Span<float> output)
    {
        var weights = new double[position + 1];
        var denominator = 0.0;
        for (var current = 0; current <= position; current++)
        {
            if (valid[current])
            {
                weights[current] = Math.Exp(scores[current] - maximum);
                denominator += weights[current];
            }
        }

        for (var coordinate = 0; coordinate < dimension; coordinate++)
        {
            var sum = 0.0;
            for (var current = 0; current <= position; current++)
            {
                if (valid[current])
                {
                    var offset = ((current * kvHeads) + kvHead) * dimension;
                    sum += weights[current] * values[offset + coordinate];
                }
            }

            var normalized = sum / denominator;
            if (!double.IsFinite(normalized) || Math.Abs(normalized) > float.MaxValue)
            {
                throw new ReferenceNumericalException(
                    ReferenceNumericalFailure.Fp32Overflow,
                    "Causal attention output exceeds the finite FP32 range.");
            }

            output[coordinate] = (float)normalized;
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

    private static ReferenceNumericalException NonFinite() => new(
        ReferenceNumericalFailure.NonFiniteInput,
        "Causal attention has a non-finite active operand.");
}
