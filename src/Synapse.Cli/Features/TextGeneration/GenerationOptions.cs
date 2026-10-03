using System.Globalization;
using ManagedCode.Synapse.Cli.Features.LayerDrop;
using ManagedCode.Synapse.Cli.Features.Speculation;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Cli.Features.TextGeneration;

/// <summary>Parsed <c>synapse generate</c> options; <see langword="null"/> from <see cref="Parse"/> means usage error.</summary>
internal sealed record GenerationOptions(
    string ModelPath,
    int[] Tokens,
    int MaximumTokens,
    int ContextSize,
    int Threads,
    KernelBackend Backend,
    int ConcurrentRequests,
    RopeScaling? RopeScaling,
    KvCachePrecision KvCachePrecision = KvCachePrecision.Fp32,
    KvPageActivation? KvPages = null,
    LayerDropProfile? LayerDrop = null,
    SpeculationArguments? Speculation = null)
{
    public static string Usage =>
        "Usage: synapse generate --model <model.synapse> (--tokens <id,id,...> | --tokens-file <path>) " +
        "[--max-tokens <count>] [--context-size <count>] [--threads <count>] " +
        $"[--backend <{KernelBackendNames.Usage}>] [--concurrent-requests <count>] " +
        "[--rope-scaling yarn:<factor>:<trained-context>] [--kv-precision f32|f16] " +
        "[--kv-pages <budget>:<window>[:p16|p32|p64][:random[:seed]]] " + LayerDropArguments.Usage + " " + SpeculationArguments.Usage;

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

        if (!values.TryGetValue("--model", out var modelPath) || !TryReadTokens(values, out var tokens))
        {
            return null;
        }

        var maximumTokens = ParsePositive(values.GetValueOrDefault("--max-tokens"), 1);
        var contextSize = ParsePositive(values.GetValueOrDefault("--context-size"), 512);
        var threads = ParsePositive(values.GetValueOrDefault("--threads"), Environment.ProcessorCount);
        var backend = KernelBackend.Managed;
        if (values.TryGetValue("--backend", out var backendName) &&
            !KernelBackendNames.TryParse(backendName, out backend))
        {
            return null;
        }

        RopeScaling? scaling = null;
        if (values.TryGetValue("--rope-scaling", out var scalingText) && !TryParseScaling(scalingText, out scaling))
        {
            return null;
        }

        var kvPrecision = values.GetValueOrDefault("--kv-precision", "f32") switch
        {
            "f32" => KvCachePrecision.Fp32,
            "f16" => KvCachePrecision.Fp16,
            _ => (KvCachePrecision?)null,
        };
        if (kvPrecision is null)
        {
            return null;
        }

        KvPageActivation? pages = null;
        if (values.TryGetValue("--kv-pages", out var pagesText) && !TryParseKvPages(pagesText, out pages))
        {
            return null;
        }

        if (!LayerDropArguments.TryParse(values, out var drop) || !SpeculationArguments.TryParse(values, out var speculation))
        {
            return null;
        }

        var concurrentRequests = ParsePositive(values.GetValueOrDefault("--concurrent-requests"), 1);
        return tokens is { Length: > 0 } && maximumTokens > 0 && contextSize > 0 && threads > 0 &&
            concurrentRequests is > 0 and <= 64
            ? new GenerationOptions(
                modelPath, tokens, maximumTokens, contextSize, threads, backend, concurrentRequests, scaling,
                kvPrecision.Value, pages, drop, speculation)
            : null;
    }

    private static bool TryReadTokens(Dictionary<string, string> values, out int[]? tokens)
    {
        tokens = null;
        var hasInline = values.TryGetValue("--tokens", out var inline);
        var hasFile = values.TryGetValue("--tokens-file", out var path);
        if (hasInline == hasFile)
        {
            return false;
        }

        if (hasInline)
        {
            tokens = ParseTokens(inline!, [',']);
            return tokens is not null;
        }

        if (!File.Exists(path))
        {
            return false;
        }

        tokens = ParseTokens(File.ReadAllText(path), [',', ' ', '\t', '\r', '\n']);
        return tokens is not null;
    }

    internal static int[]? ParseTokens(string value, char[] separators)
    {
        var parts = value.Split(separators, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var tokens = new int[parts.Length];
        for (var index = 0; index < parts.Length; index++)
        {
            if (!int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out tokens[index]))
            {
                return null;
            }
        }

        return tokens;
    }

    /// <summary>
    /// Parses <c>budget:window</c>, then optionally <c>:p16</c>, <c>:p32</c>, or <c>:p64</c> (page size), then
    /// <c>:random</c> and a seed (ADR-016).
    /// </summary>
    internal static bool TryParseKvPages(string value, out KvPageActivation? pages)
    {
        pages = null;
        var parts = new Queue<string>(value.Split(':'));
        if (parts.Count < 2 ||
            !int.TryParse(parts.Dequeue(), NumberStyles.None, CultureInfo.InvariantCulture, out var budget) ||
            !int.TryParse(parts.Dequeue(), NumberStyles.None, CultureInfo.InvariantCulture, out var window))
        {
            return false;
        }

        var size = 64;
        if (parts.TryPeek(out var page) && page is "p16" or "p32" or "p64")
        {
            size = int.Parse(parts.Dequeue()[1..], CultureInfo.InvariantCulture);
        }

        var random = parts.TryPeek(out var mode) && mode == "random" && parts.Dequeue() == "random";
        var seed = 0;
        if ((parts.Count == 1 && (!random || !int.TryParse(parts.Dequeue(), NumberStyles.None, CultureInfo.InvariantCulture, out seed))) ||
            parts.Count > 0)
        {
            return false;
        }

        pages = new KvPageActivation(budget, window)
        {
            PageTokens = size,
            Selection = random ? KvPageSelection.Random : KvPageSelection.KeyBound,
            Seed = seed,
        };
        return true;
    }

    internal static bool TryParseScaling(string value, out RopeScaling? scaling)
    {
        scaling = null;
        var parts = value.Split(':');
        if (parts is not ["yarn", var factorText, var originalText] ||
            !float.TryParse(factorText, NumberStyles.Float, CultureInfo.InvariantCulture, out var factor) ||
            !int.TryParse(originalText, NumberStyles.None, CultureInfo.InvariantCulture, out var original) ||
            !float.IsFinite(factor) || factor <= 1 || original <= 0)
        {
            return false;
        }

        scaling = RopeScaling.Yarn(factor, original);
        return true;
    }

    internal static int ParsePositive(string? value, int fallback) => value is null
            ? fallback
            : int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                ? parsed
                : -1;
}
