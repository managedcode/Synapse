using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Runtime.Features.Speculation;

/// <summary>A speculative run: the target's greedy continuation and how much of it the draft proposed.</summary>
/// <param name="Result">Prompt, committed tokens, time to first token, and wall time.</param>
/// <param name="TargetPasses">Target verification steps after the prompt; each commits at least one token.</param>
/// <param name="DraftedTokens">Tokens the draft proposed.</param>
/// <param name="AcceptedTokens">Proposed tokens that equal the target's own arg-max and were committed.</param>
public sealed record SpeculativeResult(TextGenerationResult Result, int TargetPasses, int DraftedTokens, int AcceptedTokens);

/// <summary>
/// Exact greedy speculative decoding with an external draft (ADR-020): the draft proposes tokens, the target checks
/// them all in one batched step, and only tokens the target itself would pick are committed.
/// </summary>
public static class SpeculativeDecoding
{
    /// <summary>
    /// Returns the target's greedy continuation of <paramref name="prompt"/>, using <paramref name="draft"/> to propose
    /// up to <paramref name="draftTokens"/> tokens per target pass.
    /// </summary>
    public static SpeculativeResult Generate(
        Qwen2Model target,
        Qwen2Model draft,
        IReadOnlyList<int> prompt,
        int maximumNewTokens,
        int draftTokens = 3)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumNewTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(draftTokens);
        if (ReferenceEquals(target, draft))
        {
            throw new ArgumentException("The draft must be a separate model instance with its own KV slot.", nameof(draft));
        }

        RequireCompatible(target, draft, prompt, maximumNewTokens + draftTokens);
        return target.RunDirect(targetExecutor => draft.RunDirect(draftExecutor =>
        {
            var verifier = targetExecutor as IBatchDecoder ?? throw new NotSupportedException(
                "Speculative verification needs a backend with batched steps; the reference backend has none (ADR-020).");
            ArgumentOutOfRangeException.ThrowIfGreaterThan(draftTokens + 1, verifier.LogitsRowCapacity, nameof(draftTokens));
            return new SpeculationLoop(targetExecutor, verifier, draftExecutor, draft.VocabularySize, Qwen2Model.EndOfSequenceToken)
                .Run(prompt, maximumNewTokens, draftTokens);
        }));
    }

    private static void RequireCompatible(Qwen2Model target, Qwen2Model draft, IReadOnlyList<int> prompt, int extraPositions)
    {
        if (draft.VocabularySize > target.VocabularySize ||
            draft.VocabularyDigest(draft.VocabularySize) != target.VocabularyDigest(draft.VocabularySize))
        {
            throw new NotSupportedException(
                "The draft's token IDs must be a prefix of the target's (same tokens, types, and merges; ADR-020).");
        }

        if (prompt.Count == 0 || prompt.Any(token => (uint)token >= (uint)draft.VocabularySize))
        {
            throw new ArgumentOutOfRangeException(nameof(prompt), "The prompt must be non-empty and inside the draft vocabulary.");
        }

        ArgumentOutOfRangeException.ThrowIfGreaterThan(prompt.Count + extraPositions, Math.Min(target.ContextSize, draft.ContextSize), nameof(prompt));
    }
}
