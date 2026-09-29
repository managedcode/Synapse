namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>Request limits every text generation entry point shares: a non-empty prompt of known tokens in context.</summary>
internal static class PromptLimits
{
    public static void Validate(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        bool allowEmptyOutput,
        int contextSize,
        int vocabularySize)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count == 0 || maximumNewTokens < (allowEmptyOutput ? 0 : 1) ||
            promptTokens.Count + maximumNewTokens > contextSize)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumNewTokens));
        }

        foreach (var token in promptTokens)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)token, (uint)vocabularySize, nameof(promptTokens));
        }
    }
}
