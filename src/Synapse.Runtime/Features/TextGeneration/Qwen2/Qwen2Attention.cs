namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class Qwen2Attention
{
    public static void Execute(
        Qwen2KvCache cache,
        Qwen2Scratch scratch,
        int layer,
        int position,
        int attentionHeads,
        int keyValueHeads,
        int headDimension)
    {
        var groupSize = attentionHeads / keyValueHeads;
        var scale = 1.0f / MathF.Sqrt(headDimension);
        for (var head = 0; head < attentionHeads; head++)
        {
            var query = scratch.Query.AsSpan(head * headDimension, headDimension);
            var keyValueHead = head / groupSize;
            var scores = scratch.Scores.AsSpan(0, position + 1);
            FillScores(cache, query, scores, layer, position, keyValueHead, headDimension, scale);
            SoftmaxInPlace(scores);
            AccumulateValues(cache, scratch, scores, layer, position, head, keyValueHead, headDimension);
        }
    }

    public static void ApplyRope(
        float[] values,
        int heads,
        int headDimension,
        int position,
        float ropeTheta)
    {
        var half = headDimension / 2;
        for (var head = 0; head < heads; head++)
        {
            var offset = head * headDimension;
            for (var index = 0; index < half; index++)
            {
                var angle = position / MathF.Pow(ropeTheta, 2.0f * index / headDimension);
                var cosine = MathF.Cos(angle);
                var sine = MathF.Sin(angle);
                var first = values[offset + index];
                var second = values[offset + index + half];
                values[offset + index] = (first * cosine) - (second * sine);
                values[offset + index + half] = (first * sine) + (second * cosine);
            }
        }
    }

    private static void FillScores(
        Qwen2KvCache cache,
        ReadOnlySpan<float> query,
        Span<float> scores,
        int layer,
        int position,
        int keyValueHead,
        int headDimension,
        float scale)
    {
        for (var cachedPosition = 0; cachedPosition <= position; cachedPosition++)
        {
            scores[cachedPosition] = Dot(
                query,
                cache.GetKey(layer, cachedPosition, keyValueHead, headDimension)) * scale;
        }
    }

    private static void AccumulateValues(
        Qwen2KvCache cache,
        Qwen2Scratch scratch,
        ReadOnlySpan<float> scores,
        int layer,
        int position,
        int head,
        int keyValueHead,
        int headDimension)
    {
        var destination = scratch.Attention.AsSpan(head * headDimension, headDimension);
        destination.Clear();
        for (var cachedPosition = 0; cachedPosition <= position; cachedPosition++)
        {
            var value = cache.GetValue(layer, cachedPosition, keyValueHead, headDimension);
            for (var dimension = 0; dimension < headDimension; dimension++)
            {
                destination[dimension] += scores[cachedPosition] * value[dimension];
            }
        }
    }

    private static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        var sum = 0.0f;
        for (var index = 0; index < left.Length; index++)
        {
            sum += left[index] * right[index];
        }

        return sum;
    }

    private static void SoftmaxInPlace(Span<float> values)
    {
        var maximum = values[0];
        for (var index = 1; index < values.Length; index++)
        {
            maximum = MathF.Max(maximum, values[index]);
        }

        var sum = 0.0f;
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = MathF.Exp(values[index] - maximum);
            sum += values[index];
        }

        for (var index = 0; index < values.Length; index++)
        {
            values[index] /= sum;
        }
    }
}
