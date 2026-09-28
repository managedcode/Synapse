using System.Text.Json.Nodes;

/// <summary>
/// Bounds the KV buffer that ONNX Runtime GenAI preallocates (ADR-011). Foundry packages set
/// <c>search.max_length</c> to the full context window, and <c>past_present_share_buffer</c>
/// allocates all of it at the first request. Weights and every other setting stay untouched;
/// the original configuration is kept beside the patched one and hashed into the evidence.
/// </summary>
internal static class FoundryContextBound
{
    public const string ConfigName = "genai_config.json";
    public const string OriginalName = "genai_config.original.json";

    public static async Task<int> ApplyAsync(string modelPath, int contextTokens,
        CancellationToken cancellationToken)
    {
        var config = Path.Combine(modelPath, ConfigName);
        var original = Path.Combine(modelPath, OriginalName);
        if (!File.Exists(original))
        {
            File.Copy(config, original);
        }

        var root = JsonNode.Parse(await File.ReadAllTextAsync(original, cancellationToken).ConfigureAwait(false))
            ?? throw new InvalidDataException($"{original} is empty.");
        var search = root["search"] as JsonObject
            ?? throw new InvalidDataException($"{original} has no search section.");
        var originalMaxLength = search["max_length"]?.GetValue<int>()
            ?? throw new InvalidDataException($"{original} has no search.max_length.");
        search["max_length"] = Math.Min(contextTokens, originalMaxLength);
        var temporary = config + ".tmp";
        await File.WriteAllTextAsync(temporary, root.ToJsonString(), cancellationToken).ConfigureAwait(false);
        File.Move(temporary, config, overwrite: true);
        return originalMaxLength;
    }

    public static FoundryContextState Verify(string modelPath, int contextTokens)
    {
        var original = Path.Combine(modelPath, OriginalName);
        if (!File.Exists(original))
        {
            throw new FoundryNotCachedException(
                $"The {contextTokens}-token context bound was never applied to {modelPath}; run fetch first.");
        }

        var originalMaxLength = MaxLength(original);
        var effective = MaxLength(Path.Combine(modelPath, ConfigName));
        return effective == Math.Min(contextTokens, originalMaxLength)
            ? new FoundryContextState(effective, originalMaxLength)
            : throw new FoundryNotCachedException(
                $"{modelPath} has max_length {effective}, not the set's {contextTokens}; run fetch again.");
    }

    private static int MaxLength(string path) =>
        JsonNode.Parse(File.ReadAllText(path))?["search"]?["max_length"]?.GetValue<int>()
        ?? throw new InvalidDataException($"{path} has no search.max_length.");
}

internal sealed record FoundryContextState(int EffectiveMaxLength, int OriginalMaxLength);
