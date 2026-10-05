namespace ManagedCode.Synapse.Runtime.Features.Speculation;

/// <summary>Request-local measured throughput selector; zero is ordinary target decode (ADR-020).</summary>
internal sealed class AdaptiveDepth
{
    private readonly DepthCost[] _costs;
    private int _roundsSinceProbe;
    private int _probeInterval = 16;
    private int _nextProbe = 1;

    public AdaptiveDepth(int maximumDepth)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumDepth, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(maximumDepth, 7);
        _costs = [.. Enumerable.Range(0, maximumDepth + 1).Select(_ => new DepthCost())];
    }

    public IReadOnlyList<SpeculativeDepthMeasurement> Measurements => [.. _costs.Select((cost, depth) =>
        new SpeculativeDepthMeasurement(depth, cost.Rounds, cost.Accepted, cost.Committed, cost.DraftTotal, cost.VerifyTotal))];

    public int Select(int availableDepth)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(availableDepth);
        var cap = Math.Min(availableDepth, _costs.Length - 1);
        for (var sample = 0; sample < 2; sample++)
        {
            for (var depth = 0; depth <= cap; depth++)
            {
                if (_costs[depth].Samples <= sample)
                {
                    return depth;
                }
            }
        }

        if (cap > 0 && _roundsSinceProbe++ >= _probeInterval)
        {
            _roundsSinceProbe = 0;
            _probeInterval = Math.Min(128, _probeInterval * 2);
            var probe = _nextProbe % (cap + 1);
            _nextProbe = (probe + 1) % (cap + 1);
            return probe;
        }

        var best = 0;
        var baseline = _costs[0].Rate;
        var rate = baseline;
        for (var depth = 1; depth <= cap; depth++)
        {
            var candidate = _costs[depth].Rate;
            if (candidate > baseline * 1.05 && candidate > rate)
            {
                best = depth;
                rate = candidate;
            }
        }

        return best;
    }

    public void Observe(int depth, int accepted, int committed, double draftMilliseconds, double verifyMilliseconds, bool learn = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(depth);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(depth, _costs.Length);
        ArgumentOutOfRangeException.ThrowIfNegative(accepted);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(accepted, depth);
        ArgumentOutOfRangeException.ThrowIfLessThan(committed, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(committed, depth + 1);
        RequireDuration(draftMilliseconds);
        RequireDuration(verifyMilliseconds);
        var cost = _costs[depth];
        cost.Rounds++;
        cost.Accepted += accepted;
        cost.Committed += committed;
        cost.DraftTotal += draftMilliseconds;
        cost.VerifyTotal += verifyMilliseconds;
        if (learn)
        {
            var weight = cost.Samples++ == 0 ? 1 : 0.25;
            cost.ExpectedTokens += weight * (committed - cost.ExpectedTokens);
            cost.ExpectedMilliseconds += weight * (Math.Max(0.000001, draftMilliseconds + verifyMilliseconds) - cost.ExpectedMilliseconds);
        }
    }

    private static void RequireDuration(double duration)
    {
        if (!double.IsFinite(duration) || duration < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(duration), "A measured duration must be finite and nonnegative.");
        }
    }

    private sealed class DepthCost
    {
        public int Samples { get; set; }
        public int Rounds { get; set; }
        public int Accepted { get; set; }
        public int Committed { get; set; }
        public double DraftTotal { get; set; }
        public double VerifyTotal { get; set; }
        public double ExpectedTokens { get; set; }
        public double ExpectedMilliseconds { get; set; }
        public double Rate => Samples == 0 ? 0 : ExpectedTokens / ExpectedMilliseconds;
    }
}
