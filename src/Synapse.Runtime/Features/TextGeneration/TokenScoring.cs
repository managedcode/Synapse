namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// Teacher-forced scoring over any decoder executor (ADR-015). Batch decoders score many positions per step on the
/// direct slot; the reference executor prefills to the first scored position and decodes one position at a time.
/// </summary>
internal static class TokenScoring
{
    public static TokenScores Score(
        IDecoderExecutor executor,
        IReadOnlyList<int> tokens,
        int firstScoredPosition,
        Action<int>? evaluated)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var count = tokens.Count - 1 - firstScoredPosition;
        var losses = new double[count];
        var greedy = new int[count];
        if (executor is IBatchDecoder batch)
        {
            ScoreBatched(batch, tokens, firstScoredPosition, losses, greedy, evaluated);
        }
        else
        {
            ScoreIncremental(executor, tokens, firstScoredPosition, losses, greedy, evaluated);
        }

        return new TokenScores(firstScoredPosition, losses, greedy, timer.Elapsed);
    }

    /// <summary>First index of the largest logit.</summary>
    public static int ArgMax(ReadOnlySpan<float> logits) => GreedySampling.ArgMax(logits);

    /// <summary>
    /// <c>-(logit[target] - max - log Σ expf(logit - max))</c> with an FP64 sum, the llama-perplexity expression.
    /// </summary>
    public static double NegativeLogLikelihood(ReadOnlySpan<float> logits, int target)
    {
        var maximum = logits[0];
        for (var index = 1; index < logits.Length; index++)
        {
            maximum = Math.Max(maximum, logits[index]);
        }

        var sum = 0.0;
        foreach (var logit in logits)
        {
            sum += MathF.Exp(logit - maximum);
        }

        return -((double)(logits[target] - maximum) - Math.Log(sum));
    }

    private static void ScoreBatched(
        IBatchDecoder batch,
        IReadOnlyList<int> tokens,
        int first,
        double[] losses,
        int[] greedy,
        Action<int>? evaluated)
    {
        var step = new BatchToken[batch.StepTokenCapacity];
        var end = tokens.Count - 1;
        for (var position = 0; position < end;)
        {
            var scored = position >= first;
            var count = scored
                ? Math.Min(end - position, Math.Min(batch.LogitsRowCapacity, step.Length))
                : Math.Min(first - position, step.Length);
            for (var index = 0; index < count; index++)
            {
                step[index] = new BatchToken(0, tokens[position + index], position + index, scored ? index : -1);
            }

            batch.Forward(step.AsSpan(0, count), promptStart: scored ? count : 0);
            if (scored)
            {
                var start = position;
                _ = Parallel.For(0, count, row =>
                {
                    var logits = batch.GetLogits(row);
                    losses[start - first + row] = NegativeLogLikelihood(logits, tokens[start + row + 1]);
                    greedy[start - first + row] = ArgMax(logits);
                });
            }

            position += count;
            evaluated?.Invoke(position);
        }
    }

    private static void ScoreIncremental(
        IDecoderExecutor executor,
        IReadOnlyList<int> tokens,
        int first,
        double[] losses,
        int[] greedy,
        Action<int>? evaluated)
    {
        var logits = executor.Prefill([.. tokens.Take(first + 1)]).Span;
        evaluated?.Invoke(first + 1);
        for (var index = 0; index < losses.Length; index++)
        {
            losses[index] = NegativeLogLikelihood(logits, tokens[first + index + 1]);
            greedy[index] = ArgMax(logits);
            if (index + 1 < losses.Length)
            {
                logits = executor.Decode(tokens[first + index + 1], first + index + 1).Span;
                evaluated?.Invoke(first + index + 2);
            }
        }
    }
}
