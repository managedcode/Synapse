internal static class MemoryEvaluationComparisons
{
    public static MemoryQualityComparison[] Build(IReadOnlyList<MemoryPhase> phases)
    {
        var comparisons = new List<MemoryQualityComparison>();
        foreach (var phase in phases.Where(phase => phase.Name is "short-first" or "long" or "short-after-long"))
        {
            var freshName = phase.Name == "long" ? "fresh-long" : "fresh-short";
            var fresh = phases.SingleOrDefault(phase => phase.Name == freshName)?.Generation;
            var generation = phase.Generation;
            if (fresh is null || generation is null)
            {
                continue;
            }

            var left = phases.SingleOrDefault(candidate => candidate.Name == phase.Name + "-score")?.Score;
            var right = phases.SingleOrDefault(candidate => candidate.Name == freshName + "-score")?.Score;
            var identical = generation.PromptTokenIdsSha256 == fresh.PromptTokenIdsSha256;
            var aligned = identical && left is not null && right is not null && left.FirstScoredPosition == right.FirstScoredPosition &&
                left.NegativeLogLikelihoods.Length == right.NegativeLogLikelihoods.Length && left.GreedyTokens.Length == right.GreedyTokens.Length;
            var (ratio, ratioStatus) = aligned ? OptimizationScoreFactory.Exponentiate(left!.LogPerplexity - right!.LogPerplexity)
                : (null, null);
            comparisons.Add(new MemoryQualityComparison(phase.Name, freshName, identical,
                generation.GeneratedTokens.AsSpan().SequenceEqual(fresh.GeneratedTokens),
                aligned ? left!.NegativeLogLikelihoods.Zip(right!.NegativeLogLikelihoods, (a, b) => Math.Abs(a - b)).Average() : null,
                aligned ? (double)left!.GreedyTokens.Zip(right!.GreedyTokens, (a, b) => a == b ? 1 : 0).Sum() / left.GreedyTokens.Length : null,
                ratio, ratioStatus));
        }

        return [.. comparisons];
    }
}
