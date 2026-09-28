using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Cli.Features.LongContext;

/// <summary>Parsed <c>synapse score</c> options; <see langword="null"/> from <see cref="Parse"/> means usage error.</summary>
internal sealed record ScoreOptions(
    string ModelPath,
    string? TextFile,
    string? TokensFile,
    int ContextSize,
    int? Chunks,
    int Threads,
    KernelBackend Backend,
    RopeScaling? RopeScaling,
    KvCachePrecision KvCachePrecision,
    int ScoringRows,
    bool ParseSpecialTokens,
    string? ScoresOutput,
    KvPageActivation? KvPages = null,
    int? FirstScored = null)
{
    public static string Usage =>
        "Usage: synapse score --model <model.gguf> (--text-file <path> | --tokens-file <path>) --context-size <n> " +
        $"[--chunks <n>] [--threads <n>] [--backend <{KernelBackendNames.Usage}>] " +
        "[--rope-scaling yarn:<factor>:<trained-context>] [--kv-precision f32|f16] [--scoring-rows <1..512>] " +
        "[--parse-special] [--scores-output <file.json>] [--kv-pages <budget>:<window>[:p16|p32|p64][:random[:seed]]] " +
        "[--first-scored <position>]";

    public static ScoreOptions? Parse(IReadOnlyList<string> arguments)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var parseSpecial = false;
        for (var index = 0; index < arguments.Count; index++)
        {
            if (arguments[index] == "--parse-special")
            {
                parseSpecial = true;
            }
            else if (index + 1 < arguments.Count && arguments[index].StartsWith("--", StringComparison.Ordinal))
            {
                values[arguments[index]] = arguments[++index];
            }
            else
            {
                return null;
            }
        }

        var textFile = values.GetValueOrDefault("--text-file");
        var tokensFile = values.GetValueOrDefault("--tokens-file");
        if (!values.TryGetValue("--model", out var model) || (textFile is null) == (tokensFile is null) ||
            !File.Exists(textFile ?? tokensFile) || !TryParseRuntime(values, out var backend, out var scaling, out var kv))
        {
            return null;
        }

        KvPageActivation? pages = null;
        if (values.TryGetValue("--kv-pages", out var pagesText) && !GenerationOptions.TryParseKvPages(pagesText, out pages))
        {
            return null;
        }

        var context = GenerationOptions.ParsePositive(values.GetValueOrDefault("--context-size"), -1);
        var chunks = values.TryGetValue("--chunks", out var chunkText) ? GenerationOptions.ParsePositive(chunkText, -1) : (int?)null;
        var threads = GenerationOptions.ParsePositive(values.GetValueOrDefault("--threads"), Environment.ProcessorCount);
        var rows = GenerationOptions.ParsePositive(values.GetValueOrDefault("--scoring-rows"), 64);
        var first = values.TryGetValue("--first-scored", out var firstText) ? GenerationOptions.ParsePositive(firstText, -1) : context / 2;
        return context >= 4 && chunks is null or > 0 && threads > 0 && rows is > 0 and <= 512 && first >= 0 && first <= context - 2
            ? new ScoreOptions(model, textFile, tokensFile, context, chunks, threads, backend, scaling, kv, rows,
                parseSpecial, values.GetValueOrDefault("--scores-output"), pages, first)
            : null;
    }

    private static bool TryParseRuntime(
        Dictionary<string, string> values,
        out KernelBackend backend,
        out RopeScaling? scaling,
        out KvCachePrecision kvPrecision)
    {
        backend = KernelBackend.Managed;
        scaling = null;
        kvPrecision = KvCachePrecision.Fp32;
        if ((values.TryGetValue("--backend", out var name) && !KernelBackendNames.TryParse(name, out backend)) ||
            (values.TryGetValue("--rope-scaling", out var text) && !GenerationOptions.TryParseScaling(text, out scaling)))
        {
            return false;
        }

        switch (values.GetValueOrDefault("--kv-precision", "f32"))
        {
            case "f32":
                return true;
            case "f16":
                kvPrecision = KvCachePrecision.Fp16;
                return true;
            default:
                return false;
        }
    }
}
