using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>Text to token IDs and back for a loaded model's own vocabulary (ADR-014).</summary>
public interface ITextTokenizer
{
    /// <summary>Tokens in the vocabulary.</summary>
    int VocabularySize { get; }

    /// <summary>Encodes text; with <paramref name="parseSpecialTokens"/> control tokens such as <c>&lt;|im_end|&gt;</c> are matched whole.</summary>
    IReadOnlyList<int> Encode(string text, bool parseSpecialTokens = true);

    /// <summary>Decodes token IDs to UTF-8 text; control tokens are rendered only when asked.</summary>
    string Decode(IReadOnlyList<int> tokens, bool includeSpecialTokens = false);
}

/// <summary>Loads repo-owned tokenizers from model files.</summary>
public static class TextTokenizers
{
    /// <summary>Loads tokenizer metadata from a separately compiled .synapse package.</summary>
    public static ITextTokenizer FromModel(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!string.Equals(Path.GetExtension(modelPath), ".synapse", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException("Tokenizer runtime requires .synapse. Convert first with synapse model compile --source <model.gguf> --output <model.synapse>.");
        }

        return FromGguf(modelPath);
    }
    /// <summary>Reads the byte-level BPE vocabulary, merges, and token types from a GGUF file.</summary>
    public static ITextTokenizer FromGguf(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        using var file = GgufFile.Open(modelPath);
        return FromGguf(file);
    }

    internal static ITextTokenizer FromGguf(GgufFile file)
    {
        RequireSupported(
            file.Metadata.TryGetValue("tokenizer.ggml.model", out var model) ? model as string : null,
            file.Metadata.TryGetValue("tokenizer.ggml.pre", out var pre) ? pre as string : null);
        return new ByteLevelBpeTokenizer(
            file.ReadStringArray("tokenizer.ggml.tokens"),
            file.ReadInt32Array("tokenizer.ggml.token_type"),
            file.ReadStringArray("tokenizer.ggml.merges"));
    }

    /// <summary>Fails for any tokenizer the repo does not implement; nothing falls back to an approximation.</summary>
    public static void RequireSupported(string? model, string? preTokenizer)
    {
        if (!string.Equals(model, "gpt2", StringComparison.Ordinal) ||
            !string.Equals(preTokenizer, "qwen2", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"Tokenizer '{model ?? "missing"}' with pre-tokenizer '{preTokenizer ?? "missing"}' is not implemented; " +
                "the repo-owned tokenizer supports byte-level BPE with the qwen2 pre-tokenizer.");
        }
    }
}

/// <summary>One chat turn.</summary>
/// <param name="Role">Role such as <c>system</c>, <c>user</c>, or <c>assistant</c>.</param>
/// <param name="Content">Plain message text.</param>
public sealed record ChatMessage(string Role, string Content);

/// <summary>Chat templates rendered as text with explicit special tokens.</summary>
public static class ChatTemplates
{
    /// <summary>Qwen2 ChatML: an optional system turn, the messages, and optionally the assistant header.</summary>
    public static string Qwen(string? system, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var text = new System.Text.StringBuilder();
        if (system is not null)
        {
            _ = text.Append("<|im_start|>system\n").Append(system).Append("<|im_end|>\n");
        }

        foreach (var message in messages)
        {
            _ = text.Append("<|im_start|>").Append(message.Role).Append('\n').Append(message.Content).Append("<|im_end|>\n");
        }

        if (addGenerationPrompt)
        {
            _ = text.Append("<|im_start|>assistant\n");
        }

        return text.ToString();
    }
}
