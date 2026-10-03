internal static class OptimizationComparisons
{
    public static OptimizationComparison[] Build(string baselineProfile, IReadOnlyList<OptimizationSample> samples)
    {
        var comparisons = new List<OptimizationComparison>();
        foreach (var group in samples.GroupBy(sample => (sample.CaseIndex, sample.Round, sample.Warmup)))
        {
            var baseline = group.SingleOrDefault(sample => sample.ProfileId == baselineProfile && sample.ExitCode == 0)?.Result;
            if (baseline is null)
            {
                continue;
            }

            foreach (var candidate in group.Where(sample => sample.ProfileId != baselineProfile && sample.ExitCode == 0 && sample.Result is not null))
            {
                comparisons.Add(Compare(group.Key.CaseIndex, group.Key.Round, group.Key.Warmup, baselineProfile, baseline, candidate));
            }
        }

        return [.. comparisons];
    }

    private static OptimizationComparison Compare(int caseIndex, int round, bool warmup, string baselineProfile,
        OptimizationWorkerResult baseline, OptimizationSample candidate)
    {
        var actual = candidate.Result!;
        var identical = baseline.PromptTokenIdsSha256 == actual.PromptTokenIdsSha256;
        var left = baseline.Score;
        var right = actual.Score;
        var aligned = identical && left is not null && right is not null && left.FirstScoredPosition == right.FirstScoredPosition &&
            left.NegativeLogLikelihoods.Length == right.NegativeLogLikelihoods.Length && left.GreedyTokens.Length == right.GreedyTokens.Length;
        var (ratio, ratioStatus) = aligned ? OptimizationScoreFactory.Exponentiate(right!.LogPerplexity - left!.LogPerplexity)
            : (null, null);
        return new OptimizationComparison(caseIndex, round, warmup, baselineProfile, candidate.ProfileId, identical,
            aligned ? left!.NegativeLogLikelihoods.Zip(right!.NegativeLogLikelihoods, (a, b) => Math.Abs(a - b)).Average() : null,
            aligned ? Agreement(left!.GreedyTokens, right!.GreedyTokens) : null,
            ratio, ratioStatus,
            [.. actual.Generations.Select(generation => CompareTurn(baseline.Generations.Single(turn => turn.Turn == generation.Turn), generation))]);
    }

    private static OptimizationTurnComparison CompareTurn(OptimizationGeneration baseline, OptimizationGeneration candidate) => new(
        candidate.Turn, baseline.GeneratedTokens.Length, candidate.GeneratedTokens.Length,
        baseline.GeneratedTokens.AsSpan().SequenceEqual(candidate.GeneratedTokens), Agreement(baseline.GeneratedTokens, candidate.GeneratedTokens),
        baseline.ContainsAllAnswers, candidate.ContainsAllAnswers,
        baseline.TimeToFirstTokenMilliseconds / candidate.TimeToFirstTokenMilliseconds,
        baseline.GenerationMilliseconds / candidate.GenerationMilliseconds);

    private static double Agreement(int[] left, int[] right)
    {
        var count = Math.Max(left.Length, right.Length);
        return count == 0 ? 1 : (double)left.Zip(right, (a, b) => a == b ? 1 : 0).Sum() / count;
    }
}
