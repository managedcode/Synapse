using System.Diagnostics;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>The greedy loop on one executor's direct slot once the prompt is prefilled.</summary>
internal static class GreedyDecoding
{
    /// <summary>
    /// Samples the arg-max of <paramref name="logits"/>, then decodes each sampled token at the next position until
    /// the end-of-sequence token or <paramref name="maximumNewTokens"/>. <paramref name="sampled"/> sees the running
    /// count after every token.
    /// </summary>
    public static (List<int> Generated, TimeSpan TimeToFirstToken) Continue(
        IDecoderExecutor executor,
        ReadOnlyMemory<float> logits,
        int position,
        int maximumNewTokens,
        int endOfSequenceToken,
        Stopwatch timer,
        Action<int>? sampled)
    {
        var generated = new List<int>(maximumNewTokens);
        var timeToFirstToken = TimeSpan.Zero;
        for (var index = 0; index < maximumNewTokens; index++)
        {
            var token = GreedySampling.ArgMax(logits.Span);
            generated.Add(token);
            timeToFirstToken = index == 0 ? timer.Elapsed : timeToFirstToken;
            sampled?.Invoke(generated.Count);
            if (token == endOfSequenceToken || index + 1 == maximumNewTokens)
            {
                break;
            }

            logits = executor.Decode(token, position + index);
        }

        return (generated, timeToFirstToken);
    }
}
