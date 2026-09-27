namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Token-level output from the initial Qwen2 greedy generation path.</summary>
/// <param name="PromptTokens">Input token IDs evaluated by the model.</param>
/// <param name="GeneratedTokens">Greedy output token IDs.</param>
/// <param name="Elapsed">Wall-clock time spent evaluating and generating.</param>
public sealed record Qwen2GenerationResult(
    IReadOnlyList<int> PromptTokens,
    IReadOnlyList<int> GeneratedTokens,
    TimeSpan Elapsed);
