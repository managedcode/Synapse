using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// <c>prefix-reuse</c>: two questions about one long document in one process (ADR-018). The first request prefills
/// everything; the second reuses the document's K and V and evaluates only the new question. In-process by design:
/// reuse lives in one loaded model.
/// </summary>
internal static class PrefixReuseCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index + 1 < args.Length; index += 2)
        {
            values[args[index]] = args[index + 1];
        }

        if (!values.TryGetValue("--model", out var model) || !values.TryGetValue("--haystack", out var haystack) ||
            !values.TryGetValue("--output", out var output))
        {
            await Console.Error.WriteLineAsync("Usage: prefix-reuse --model <gguf> --haystack <text> --output <file.json> " +
                "[--document-tokens 30000] [--context-size 32768] [--backend metal] [--kv-precision f16]").ConfigureAwait(false);
            return 2;
        }

        var documentTokens = int.Parse(values.GetValueOrDefault("--document-tokens", "30000"), CultureInfo.InvariantCulture);
        var context = int.Parse(values.GetValueOrDefault("--context-size", "32768"), CultureInfo.InvariantCulture);
        var tokenizer = TextTokenizers.FromGguf(model);
        var document = tokenizer.Encode(await File.ReadAllTextAsync(haystack).ConfigureAwait(false), parseSpecialTokens: false)
            .Take(documentTokens).ToArray();
        string[] questions = ["What is this text about? Answer in one sentence.", "Which programming languages appear in it?"];
        var runs = new List<PrefixReuseRun>();
        foreach (var reuse in new[] { false, true })
        {
            using var loaded = ModelLoader.Load(model, new ModelLoadOptions
            {
                ContextSize = context,
                KernelBackend = KernelBackendNames.TryParse(values.GetValueOrDefault("--backend", "metal"), out var backend)
                    ? backend
                    : KernelBackend.Metal,
                KvCachePrecision = values.GetValueOrDefault("--kv-precision", "f16") == "f32" ? KvCachePrecision.Fp32 : KvCachePrecision.Fp16,
                ReusePromptPrefix = reuse,
            });
            for (var turn = 0; turn < questions.Length; turn++)
            {
                var prompt = tokenizer.Encode(ChatTemplates.Qwen("You are a helpful assistant.",
                    [new ChatMessage("user", tokenizer.Decode(document) + "\n\n" + questions[turn])], addGenerationPrompt: true));
                var result = loaded.Generate(prompt, 32);
                runs.Add(new PrefixReuseRun(reuse, turn + 1, prompt.Count, result.ReusedPromptTokens,
                    result.TimeToFirstToken.TotalMilliseconds, result.Elapsed.TotalMilliseconds, tokenizer.Decode(result.GeneratedTokens)));
                await Console.Error.WriteLineAsync(string.Create(CultureInfo.InvariantCulture,
                    $"reuse={reuse} turn={turn + 1} prompt={prompt.Count} reused={result.ReusedPromptTokens} ttft={result.TimeToFirstToken.TotalMilliseconds:F0}ms"))
                    .ConfigureAwait(false);
            }
        }

        await File.WriteAllTextAsync(output, JsonSerializer.Serialize(
            new PrefixReuseEvidence(1, "prefix-reuse-diagnostic", DateTimeOffset.UtcNow, Path.GetFullPath(model), context, runs),
            PrefixReuseJsonContext.Default.PrefixReuseEvidence)).ConfigureAwait(false);
        return 0;
    }
}

internal sealed record PrefixReuseRun(
    bool ReusePromptPrefix,
    int Turn,
    int PromptTokens,
    int ReusedPromptTokens,
    double TimeToFirstTokenMilliseconds,
    double ElapsedMilliseconds,
    string Output);

internal sealed record PrefixReuseEvidence(
    int SchemaVersion,
    string Kind,
    DateTimeOffset RecordedAt,
    string ModelPath,
    int ContextSize,
    IReadOnlyList<PrefixReuseRun> Runs);

[JsonSerializable(typeof(PrefixReuseEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
internal sealed partial class PrefixReuseJsonContext : JsonSerializerContext;
