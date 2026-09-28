using System.Globalization;

/// <summary>Parsed <c>sweep</c> options; <see langword="null"/> from <see cref="Parse"/> means usage error.</summary>
internal sealed record SweepOptions(
    string ModelPath,
    string Haystack,
    IReadOnlyList<int> Contexts,
    IReadOnlyList<string> Subjects,
    string Output,
    string SynapseExecutable,
    string LlamaCompletion,
    string? MlxBinary,
    string? MlxModel,
    int MlxPort,
    int MaxTokens,
    int Warmups,
    int Measurements,
    int Threads,
    int CpuMaxContext,
    int Seed,
    string Reference,
    string? PromptsDirectory = null)
{
    public const string Usage =
        "Usage: sweep --model <gguf> --haystack <text> --contexts <n,n,...> --output <file.json> " +
        "--subjects <synapse:<backend>:<f32|f16>,llamacpp:<metal|cpu>:<f32|f16|q8_0>,mlx,...> " +
        "--synapse-executable <path> [--llama-completion <path>] [--mlx-binary <SwiftLM> --mlx-model <dir>] " +
        "[--mlx-port 15416] [--max-tokens 128] [--warmups 1] [--measurements 2] [--threads 8] " +
        "[--cpu-max-context 8192] [--seed 20260928] [--reference llamacpp-metal/f16] [--prompts-dir <dir>]";

    public static SweepOptions? Parse(IReadOnlyList<string> args)
    {
        if (args.Count % 2 != 0)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Count; index += 2)
        {
            if (!args[index].StartsWith("--", StringComparison.Ordinal))
            {
                return null;
            }

            values[args[index]] = args[index + 1];
        }

        try
        {
            return Create(values);
        }
        catch (Exception exception) when (exception is FormatException or KeyNotFoundException)
        {
            return null;
        }
    }

    private static SweepOptions? Create(Dictionary<string, string> values)
    {
        int Count(string name, int fallback)
        {
            return values.TryGetValue(name, out var text) ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;
        }

        var contexts = values["--contexts"].Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToArray();
        var subjects = values["--subjects"].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var options = new SweepOptions(values["--model"], values["--haystack"], contexts, subjects, values["--output"],
            values["--synapse-executable"], values.GetValueOrDefault("--llama-completion", "llama-completion"),
            values.GetValueOrDefault("--mlx-binary"), values.GetValueOrDefault("--mlx-model"), Count("--mlx-port", 15416),
            Count("--max-tokens", 128), Count("--warmups", 1), Count("--measurements", 2), Count("--threads", 8),
            Count("--cpu-max-context", 8192), Count("--seed", 20260928),
            values.GetValueOrDefault("--reference", "llamacpp-metal/f16"), values.GetValueOrDefault("--prompts-dir"));
        return contexts.Length > 0 && contexts.All(context => context > options.MaxTokens + 256) && subjects.Length > 0 &&
            options.MaxTokens > 0 && options.Warmups >= 0 && options.Measurements > 0
            ? options
            : null;
    }
}
