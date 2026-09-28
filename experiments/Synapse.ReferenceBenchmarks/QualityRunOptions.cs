using System.Globalization;

/// <summary>Parsed <c>quality</c> options; <see langword="null"/> from <see cref="Parse"/> means usage error.</summary>
internal sealed record QualityRunOptions(
    string ModelPath,
    string Haystack,
    IReadOnlyList<int> Lengths,
    IReadOnlyList<double> Depths,
    IReadOnlyList<string> Tasks,
    IReadOnlyList<string> Subjects,
    int ContextSize,
    string Output,
    string? RopeScaling,
    int Seed,
    int Threads,
    string SynapseExecutable,
    string? LlamaCompletion,
    string? LlamaTokenize,
    string? MlxBinary,
    string? MlxModel,
    int MlxPort)
{
    public const string Usage =
        "Usage: quality --model <gguf> --haystack <text> --lengths <n,n,...> --context-size <n> --output <file.json> " +
        "--subjects <synapse:<backend>:<f32|f16>,llamacpp:<metal|cpu>,mlx,...> --synapse-executable <path> " +
        "[--depths 0.1,0.5,0.9] [--tasks needle,multikey,vartrack] [--rope-scaling yarn:<factor>:<trained>] " +
        "[--seed 20260928] [--threads 8] [--llama-completion <path>] [--llama-tokenize <path>] " +
        "[--mlx-binary <SwiftLM>] [--mlx-model <dir>] [--mlx-port 15415]";

    public static QualityRunOptions? Parse(IReadOnlyList<string> args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index + 1 < args.Count; index += 2)
        {
            if (!args[index].StartsWith("--", StringComparison.Ordinal))
            {
                return null;
            }

            values[args[index]] = args[index + 1];
        }

        string? Get(string name)
        {
            return values.GetValueOrDefault(name);
        }

        var lengths = List(Get("--lengths"), value => int.Parse(value, CultureInfo.InvariantCulture));
        var depths = List(Get("--depths") ?? "0.1,0.5,0.9", value => double.Parse(value, CultureInfo.InvariantCulture));
        var tasks = List(Get("--tasks") ?? string.Join(',', QualityTaskFactory.TaskNames), value => value);
        var subjects = List(Get("--subjects"), value => value);
        if (args.Count % 2 != 0 || Get("--model") is not { } model || Get("--haystack") is not { } haystack ||
            Get("--output") is not { } output || Get("--synapse-executable") is not { } synapse ||
            !int.TryParse(Get("--context-size"), CultureInfo.InvariantCulture, out var context) ||
            lengths is not { Count: > 0 } || depths is null || tasks is null || subjects is not { Count: > 0 } ||
            tasks.Any(task => !QualityTaskFactory.TaskNames.Contains(task)) || lengths.Any(length => length > context))
        {
            return null;
        }

        return new QualityRunOptions(model, haystack, lengths, depths, tasks, subjects, context, output,
            Get("--rope-scaling"), int.Parse(Get("--seed") ?? "20260928", CultureInfo.InvariantCulture),
            int.Parse(Get("--threads") ?? "8", CultureInfo.InvariantCulture), synapse,
            Get("--llama-completion") ?? "llama-completion", Get("--llama-tokenize") ?? "llama-tokenize",
            Get("--mlx-binary"), Get("--mlx-model"), int.Parse(Get("--mlx-port") ?? "15415", CultureInfo.InvariantCulture));
    }

    private static List<T>? List<T>(string? text, Func<string, T> parse)
    {
        if (text is null)
        {
            return null;
        }

        try
        {
            return [.. text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(parse)];
        }
        catch (FormatException)
        {
            return null;
        }
    }
}
