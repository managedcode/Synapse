using ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>Finite log-space score evidence and explicit exponent limits (ADR-025).</summary>
internal static class OptimizationScoreFactory
{
    public static OptimizationScore Create(TokenScores scores)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (scores.NegativeLogLikelihoods.Count == 0 ||
            scores.NegativeLogLikelihoods.Count != scores.GreedyTokens.Count)
        {
            throw new InvalidDataException("Scoring needs nonempty aligned likelihood and greedy-token rows.");
        }

        if (scores.NegativeLogLikelihoods.Any(value => !double.IsFinite(value)))
        {
            throw new InvalidDataException("Scoring produced a nonfinite negative log-likelihood.");
        }

        var mean = scores.NegativeLogLikelihoods.Average();
        var (perplexity, status) = Exponentiate(mean);
        return new OptimizationScore(scores.FirstScoredPosition, [.. scores.NegativeLogLikelihoods], [.. scores.GreedyTokens],
            mean, mean, perplexity, status, scores.Elapsed.TotalMilliseconds);
    }

    public static (double? Value, string Status) Exponentiate(double logarithm)
    {
        if (!double.IsFinite(logarithm))
        {
            throw new InvalidDataException("Scoring produced a nonfinite logarithm.");
        }

        var value = Math.Exp(logarithm);
        if (double.IsPositiveInfinity(value))
        {
            return (null, "overflow");
        }

        return value == 0 ? (null, "underflow") : (value, "finite");
    }
}
