namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// One token of a batched forward step: the owning KV slot, the token ID, its position in that slot, and the
/// logits row to fill, or -1 when no logits are needed (ADR-007).
/// </summary>
internal readonly record struct BatchToken(int Slot, int Token, int Position, int LogitsRow);

/// <summary>A model executor that evaluates tokens of several KV slots in one pass over the weights.</summary>
internal interface IBatchDecoder
{
    /// <summary>Maximum tokens in one step.</summary>
    int StepTokenCapacity { get; }

    /// <summary>KV slots; slot 0 is reserved for the direct synchronous path.</summary>
    int SessionSlots { get; }

    int VocabularySize { get; }

    /// <summary>Logit rows one step may request.</summary>
    int LogitsRowCapacity { get; }

    /// <summary>Sizes a newly assigned slot before its first step; the caller holds the execution gate (ADR-017).</summary>
    void Reserve(int slot, int positions);

    /// <summary>
    /// Evaluates one step. Logits rows must be exactly <c>0..k-1</c> in token order. Tokens from index
    /// <paramref name="promptStart"/> on are prompt tokens: they always attend densely, even alone in their slot (ADR-016).
    /// </summary>
    void Forward(ReadOnlySpan<BatchToken> tokens, int promptStart);

    ReadOnlySpan<float> GetLogits(int row);
}

/// <summary>What one scheduler step fed for one session; published for diagnostics and tests.</summary>
internal sealed record BatchStepEntry(
    int Slot,
    int PromptLength,
    int PrefillTokens,
    int DecodeTokens,
    int GeneratedBefore);

/// <summary>Composition of one continuous-batching step.</summary>
internal sealed record BatchStepTrace(IReadOnlyList<BatchStepEntry> Entries);
