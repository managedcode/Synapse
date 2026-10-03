using ManagedCode.Synapse.Cli.Features.LayerDrop;
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
    int? FirstScored = null,
    LayerDropProfile? LayerDrop = null,
    string Optimization = "custom")
{
    public static string Usage =>
        "Usage: synapse score --model <model.synapse> (--text-file <path> | --tokens-file <path>) --context-size <n> " +
        $"[--chunks <n>] [--threads <n>] [--backend <{KernelBackendNames.Usage}>] " +
        "[--rope-scaling yarn:<factor>:<trained-context>] [--kv-precision f32|f16] [--scoring-rows <1..512>] " +
        "[--parse-special] [--scores-output <file.json>] [--kv-pages <budget>:<window>[:p16|p32|p64][:random[:seed]]] " +
        "[--first-scored <position>] " + LayerDropArguments.Usage + " " + RuntimeOptimizationArguments.Usage;

    public static ScoreOptions? Parse(IReadOnlyList<string> arguments)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var parseSpecial = false;
        for (var index = 0; index < arguments.Count; index++)
        {
            if (arguments[index] == "--parse-special")
            {
                if (parseSpecial)
                {
                    return null;
                }

                parseSpecial = true;
            }
            else if (index + 1 < arguments.Count && RuntimeOptimizationArguments.IsScoreFlag(arguments[index]))
            {
                if (!values.TryAdd(arguments[index], arguments[++index]))
                {
                    return null;
                }
            }
            else
            {
                return null;
            }
        }

        var textFile = values.GetValueOrDefault("--text-file");
        var tokensFile = values.GetValueOrDefault("--tokens-file");
        if (!values.TryGetValue("--model", out var model) || (textFile is null) == (tokensFile is null) ||
            !File.Exists(textFile ?? tokensFile) || !RuntimeOptimizationArguments.TryParseRuntime(values, out var backend, out var scaling, out var kv) ||
            !RuntimeOptimizationArguments.TryResolve(values, ref backend, out var optimization))
        {
            return null;
        }

        KvPageActivation? pages = null;
        if (values.TryGetValue("--kv-pages", out var pagesText) && !GenerationOptions.TryParseKvPages(pagesText, out pages))
        {
            return null;
        }

        if (!LayerDropArguments.TryParse(values, out var drop))
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
                parseSpecial, values.GetValueOrDefault("--scores-output"), pages, first, drop, optimization)
            : null;
    }
}
