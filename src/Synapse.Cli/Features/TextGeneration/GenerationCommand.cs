using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Cli.Features.TextGeneration;

internal static class GenerationCommand
{
    public static int Run(IReadOnlyList<string> arguments)
    {
        var options = GenerationOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine(
                "Usage: synapse generate --model <model.gguf> --tokens <id,id,...> " +
                "[--max-tokens <count>] [--context-size <count>]");
            return 2;
        }

        try
        {
            var loadTimer = Stopwatch.StartNew();
            using var model = Qwen2Model.Load(options.ModelPath, options.ContextSize);
            loadTimer.Stop();
            var result = model.Generate(options.Tokens, options.MaximumTokens);
            var output = new GenerationOutput(
                "synapse-managed-qwen2-q8_0",
                Path.GetFullPath(options.ModelPath),
                result.PromptTokens,
                result.GeneratedTokens,
                loadTimer.Elapsed.TotalMilliseconds,
                result.Elapsed.TotalMilliseconds,
                result.Elapsed.TotalSeconds == 0
                    ? null
                    : result.GeneratedTokens.Count / result.Elapsed.TotalSeconds);
            Console.WriteLine(JsonSerializer.Serialize(output, GenerationJsonContext.Default.GenerationOutput));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private sealed record GenerationOptions(
        string ModelPath,
        int[] Tokens,
        int MaximumTokens,
        int ContextSize)
    {
        public static GenerationOptions? Parse(IReadOnlyList<string> arguments)
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var index = 0; index < arguments.Count; index += 2)
            {
                if (index + 1 >= arguments.Count ||
                    !arguments[index].StartsWith("--", StringComparison.Ordinal))
                {
                    return null;
                }

                values[arguments[index]] = arguments[index + 1];
            }

            if (!values.TryGetValue("--model", out var modelPath) ||
                !values.TryGetValue("--tokens", out var tokenText))
            {
                return null;
            }

            var tokens = ParseTokens(tokenText);
            var maximumTokens = ParsePositive(values.GetValueOrDefault("--max-tokens"), 1);
            var contextSize = ParsePositive(values.GetValueOrDefault("--context-size"), 512);
            return tokens is { Length: > 0 } && maximumTokens > 0 && contextSize > 0
                ? new GenerationOptions(modelPath, tokens, maximumTokens, contextSize)
                : null;
        }

        private static int[]? ParseTokens(string value)
        {
            var parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var tokens = new int[parts.Length];
            for (var index = 0; index < parts.Length; index++)
            {
                if (!int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out tokens[index]) ||
                    tokens[index] < 0)
                {
                    return null;
                }
            }

            return tokens;
        }

        private static int ParsePositive(string? value, int fallback) => value is null
                ? fallback
                : int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                    ? parsed
                    : -1;
    }
}

internal sealed record GenerationOutput(
    string Subject,
    string ModelPath,
    IReadOnlyList<int> PromptTokens,
    IReadOnlyList<int> GeneratedTokens,
    double LoadMilliseconds,
    double GenerationMilliseconds,
    double? TokensPerSecond);

[JsonSerializable(typeof(GenerationOutput))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class GenerationJsonContext : JsonSerializerContext;
