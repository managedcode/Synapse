using ManagedCode.Synapse.Contracts.Features.GraphExecution;

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

    /// <summary>Runs bounded greedy generation from token IDs.</summary>
    TextGenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens);

    /// <summary>
    /// Thread-safe greedy generation. Concurrent calls share batched forward steps on optimized backends and
    /// produce the same tokens as independent calls (ADR-007).
    /// </summary>
    Task<TextGenerationResult> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        CancellationToken cancellationToken);
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
    TimeSpan Elapsed);
