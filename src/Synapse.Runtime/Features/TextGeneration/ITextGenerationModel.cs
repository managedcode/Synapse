using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>A loaded local model with a verified portable graph.</summary>
public interface ITextGenerationModel : IDisposable
{
    /// <summary>Canonical architecture family selected by the source adapter.</summary>
    string Architecture { get; }

    /// <summary>Concrete runtime and numerical profile.</summary>
    string RuntimeProfile { get; }

    /// <summary>Concrete kernel and instruction-set implementation selected at load.</summary>
    string KernelImplementation { get; }

    /// <summary>Verified family-independent model graph.</summary>
    ModelGraph Graph { get; }

    /// <summary>Returns the model's cached tokenizer from its verified mapping; unsupported tokenizers fail explicitly.</summary>
    ITextTokenizer CreateTokenizer();

    /// <summary>Runs bounded greedy generation from token IDs.</summary>
    TextGenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens);

    /// <summary>Runs bounded greedy generation and reports prompt and output progress synchronously.</summary>
    TextGenerationResult Generate(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        IProgress<GenerationProgress>? progress);

    /// <summary>
    /// Thread-safe greedy generation. Concurrent calls share batched forward steps on optimized backends and
    /// produce the same tokens as independent calls (ADR-007).
    /// </summary>
    Task<TextGenerationResult> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        CancellationToken cancellationToken);

    /// <summary>
    /// Teacher-forced scoring in a fresh direct session (ADR-015): evaluates <paramref name="tokens"/> from position 0
    /// and, for every position from <paramref name="firstScoredPosition"/> to <c>n-2</c>, returns the negative
    /// log-likelihood of the next token and the greedy token.
    /// </summary>
    TokenScores Score(IReadOnlyList<int> tokens, int firstScoredPosition, IProgress<GenerationProgress>? progress);
}

/// <summary>Token-level output shared by every causal text family.</summary>
/// <param name="PromptTokens">Input token IDs evaluated by the model.</param>
/// <param name="GeneratedTokens">Greedy output token IDs.</param>
/// <param name="TimeToFirstToken">Wall time through the first sampled output token.</param>
/// <param name="Elapsed">Wall-clock time spent evaluating and generating.</param>
public sealed record TextGenerationResult(
    IReadOnlyList<int> PromptTokens,
    IReadOnlyList<int> GeneratedTokens,
    TimeSpan TimeToFirstToken,
    TimeSpan Elapsed)
{
    /// <summary>Prompt tokens whose K and V were reused instead of evaluated (ADR-018).</summary>
    public int ReusedPromptTokens { get; init; }
}

/// <summary>Progress of one generation call, reported after each prompt chunk and each generated token.</summary>
/// <param name="EvaluatedPromptTokens">Prompt tokens already evaluated.</param>
/// <param name="PromptTokens">Total prompt tokens.</param>
/// <param name="GeneratedTokens">Output tokens sampled so far.</param>
/// <param name="Elapsed">Wall time since the call started.</param>
public sealed record GenerationProgress(
    int EvaluatedPromptTokens,
    int PromptTokens,
    int GeneratedTokens,
    TimeSpan Elapsed);

/// <summary>Teacher-forced scores of one token sequence (ADR-015).</summary>
/// <param name="FirstScoredPosition">Position whose logits predict the first scored token.</param>
/// <param name="NegativeLogLikelihoods">
/// Entry <c>i</c> is <c>-log P(tokens[p+1] | tokens[0..p])</c> for <c>p = FirstScoredPosition + i</c>.
/// </param>
/// <param name="GreedyTokens">Entry <c>i</c> is the arg-max token at position <c>FirstScoredPosition + i</c>.</param>
/// <param name="Elapsed">Wall time spent evaluating and scoring.</param>
public sealed record TokenScores(
    int FirstScoredPosition,
    IReadOnlyList<double> NegativeLogLikelihoods,
    IReadOnlyList<int> GreedyTokens,
    TimeSpan Elapsed);
