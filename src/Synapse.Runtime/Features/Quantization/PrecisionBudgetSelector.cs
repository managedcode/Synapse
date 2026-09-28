using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.Quantization;

/// <summary>
/// Greedy budgeted static precision per tensor (TASK-QNT-003). Every tensor starts at its
/// first supported encoding. While the total exceeds the budget, the next demotion with
/// the smallest distortion increase per saved byte is applied; ties go to the lower
/// tensor identity. Important tensors therefore keep precision, and unimportant ones
/// drop first, down to ternary where allowed. The result is a heuristic profile that
/// still needs held-out quality evidence before it can be a default.
/// The demotion order does not depend on the budget, so a larger budget only stops the
/// same sequence earlier: raising the budget promotes the most important demoted tensors
/// first and never lowers any tensor's precision.
/// </summary>
public static class PrecisionBudgetSelector
{
    private const string ProfileHashVersion = "synapse.precision-profile.v1";

    /// <summary>Selects one encoding per tensor under <paramref name="policy"/>.</summary>
    public static PrecisionPlan Select(IReadOnlyList<PrecisionCandidate> candidates, PrecisionPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(policy.BudgetBytes);
        var states = CreateStates(candidates, policy);
        var total = states.Sum(state => state.Current.Bytes);
        if (total > policy.BudgetBytes && !policy.AllowApproximation)
        {
            throw new PrecisionBudgetException(
                PrecisionBudgetFailure.ApproximationNotAllowed,
                total,
                $"Source precision needs {total} bytes, over the {policy.BudgetBytes}-byte budget, and approximation is not allowed.");
        }

        if (states.Any(state => state.IsDemoted) && !policy.AllowApproximation)
        {
            throw new PrecisionBudgetException(
                PrecisionBudgetFailure.ApproximationNotAllowed,
                total,
                "The target device lacks a source encoding kernel, and approximation is not allowed.");
        }

        var minimum = states.Sum(state => state.Lowest.Bytes);
        if (minimum > policy.BudgetBytes)
        {
            throw new PrecisionBudgetException(
                PrecisionBudgetFailure.InsufficientBudget,
                minimum,
                $"The lowest allowed encodings need {minimum} bytes, over the {policy.BudgetBytes}-byte budget.");
        }

        while (total > policy.BudgetBytes)
        {
            var next = FindCheapestDemotion(states);
            total -= next.Current.Bytes - next.Next.Bytes;
            next.Demote();
        }

        return CreatePlan(states, total);
    }

    /// <summary>Lists the per-tensor encoding changes needed to move from one profile to another.</summary>
    public static IReadOnlyList<PrecisionChange> Diff(PrecisionPlan from, PrecisionPlan to)
    {
        ArgumentNullException.ThrowIfNull(from);
        ArgumentNullException.ThrowIfNull(to);
        var target = to.Assignments.ToDictionary(item => item.Tensor);
        if (target.Count != from.Assignments.Count || from.Assignments.Any(item => !target.ContainsKey(item.Tensor)))
        {
            throw new ArgumentException("Profiles must cover the same tensors.");
        }

        return Array.AsReadOnly(from.Assignments
            .Where(item => !string.Equals(item.EncodingId, target[item.Tensor].EncodingId, StringComparison.Ordinal))
            .Select(item => new PrecisionChange(
                item.Tensor,
                item.EncodingId,
                target[item.Tensor].EncodingId,
                target[item.Tensor].Bytes - item.Bytes))
            .ToArray());
    }

    private static CandidateState[] CreateStates(IReadOnlyList<PrecisionCandidate> candidates, PrecisionPolicy policy)
    {
        var seen = new HashSet<TensorId>();
        var states = new CandidateState[candidates.Count];
        for (var index = 0; index < candidates.Count; index++)
        {
            var candidate = candidates[index] ?? throw new ArgumentException("Precision candidates cannot be null.");
            ValidateOptions(candidate);
            if (!seen.Add(candidate.Tensor))
            {
                throw new ArgumentException($"Tensor {candidate.Tensor} appears more than once.");
            }

            var supported = candidate.Options
                .Where(option => policy.SupportedEncodings?.Contains(option.EncodingId) ?? true)
                .ToArray();
            if (supported.Length == 0)
            {
                throw new PrecisionBudgetException(
                    PrecisionBudgetFailure.UnsupportedEncoding,
                    0,
                    $"Tensor {candidate.Tensor} has no encoding supported by the target device.");
            }

            states[index] = new CandidateState(candidate, supported);
        }

        Array.Sort(states, (left, right) => left.Tensor.Value.CompareTo(right.Tensor.Value));
        return states;
    }

    private static void ValidateOptions(PrecisionCandidate candidate)
    {
        if (candidate.Options is null || candidate.Options.Count == 0)
        {
            throw new ArgumentException($"Tensor {candidate.Tensor} has no precision options.");
        }

        for (var index = 0; index < candidate.Options.Count; index++)
        {
            var option = candidate.Options[index];
            if (option is null || string.IsNullOrWhiteSpace(option.EncodingId) || option.Bytes <= 0 ||
                !double.IsFinite(option.Distortion) || option.Distortion < 0 ||
                (index > 0 && option.Bytes >= candidate.Options[index - 1].Bytes))
            {
                throw new ArgumentException(
                    $"Tensor {candidate.Tensor} options need encodings, positive strictly decreasing sizes, and finite non-negative distortion.");
            }
        }
    }

    private static CandidateState FindCheapestDemotion(CandidateState[] states)
    {
        CandidateState? best = null;
        var bestScore = double.PositiveInfinity;
        foreach (var state in states)
        {
            if (!state.CanDemote)
            {
                continue;
            }

            var score = (state.Next.Distortion - state.Current.Distortion) / (state.Current.Bytes - state.Next.Bytes);
            if (score < bestScore)
            {
                best = state;
                bestScore = score;
            }
        }

        return best ?? throw new InvalidOperationException("No demotion remains although the minimum fits the budget.");
    }

    private static PrecisionPlan CreatePlan(CandidateState[] states, long total)
    {
        var assignments = states
            .Select(state => new PrecisionAssignment(
                state.Tensor,
                state.Current.EncodingId,
                state.Current.Bytes,
                state.Current.Distortion,
                state.IsDemoted))
            .ToArray();
        return new PrecisionPlan(
            Array.AsReadOnly(assignments),
            total,
            assignments.Sum(item => item.Distortion),
            assignments.Any(item => item.IsDemoted),
            ComputeProfileHash(assignments));
    }

    private static ContentHash ComputeProfileHash(IEnumerable<PrecisionAssignment> assignments)
    {
        var canonical = new StringBuilder(ProfileHashVersion).Append('\n');
        foreach (var assignment in assignments)
        {
            _ = canonical
                .Append(assignment.Tensor.Value.ToString(CultureInfo.InvariantCulture))
                .Append(':')
                .Append(assignment.EncodingId)
                .Append('\n');
        }

        var digest = SHA256.HashData(Encoding.UTF8.GetBytes(canonical.ToString()));
        return new ContentHash(Convert.ToHexStringLower(digest));
    }

    private sealed class CandidateState(PrecisionCandidate candidate, PrecisionOption[] supported)
    {
        private int _index;

        public TensorId Tensor => candidate.Tensor;

        public PrecisionOption Current => supported[_index];

        public PrecisionOption Next => supported[_index + 1];

        public PrecisionOption Lowest => candidate.PinnedHigh ? supported[0] : supported[^1];

        public bool CanDemote => !candidate.PinnedHigh && _index + 1 < supported.Length;

        public bool IsDemoted => !string.Equals(Current.EncodingId, candidate.Options[0].EncodingId, StringComparison.Ordinal);

        public void Demote() => _index++;
    }
}
