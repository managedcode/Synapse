using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

/// <summary>An engine measured by the context sweep: one fresh process (or request) per sample.</summary>
internal interface ISweepSubject : IAsyncDisposable
{
    string Name { get; }

    string KvCache { get; }

    bool CpuOnly { get; }

    Task<SweepRun> RunAsync(QualityCase prompt, int context, CancellationToken cancellationToken);
}

/// <summary>One sample: engine-reported phase timings, whole-process memory, and the generated text.</summary>
internal sealed record SweepRun(
    int ExitCode,
    string Output,
    int? PromptTokens,
    bool? TokenIdsIdentical,
    double? TimeToFirstTokenMilliseconds,
    double? GenerationMilliseconds,
    int GeneratedTokens,
    double? DecodeTokensPerSecond,
    double WallMilliseconds,
    long? PeakFootprintBytes,
    long? PeakResidentBytes,
    string? Error);

/// <summary><c>synapse generate --tokens-file</c>; timings come from the CLI's own phase clock.</summary>
internal sealed class SynapseSweepSubject(SweepOptions options, string backend, string kv) : ISweepSubject
{
    public string Name => $"synapse-{backend}";

    public string KvCache => kv;

    public bool CpuOnly => backend is not ("metal" or "cuda");

    public async Task<SweepRun> RunAsync(QualityCase prompt, int context, CancellationToken cancellationToken)
    {
        var tokensFile = Path.Combine(Path.GetTempPath(), $"synapse-sweep-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(tokensFile, string.Join('\n', prompt.Tokens), cancellationToken).ConfigureAwait(false);
        try
        {
            var run = await QualityProcess.RunAsync(options.SynapseExecutable,
            [
                "generate", "--model", options.ModelPath, "--tokens-file", tokensFile,
                "--max-tokens", Number(options.MaxTokens), "--context-size", Number(context),
                "--backend", backend, "--kv-precision", kv, "--threads", Number(options.Threads),
            ], cancellationToken, sampleMemory: true).ConfigureAwait(false);
            if (run.ExitCode != 0)
            {
                return Failed(run);
            }

            using var json = JsonDocument.Parse(run.Output);
            var root = json.RootElement;
            var promptIds = root.GetProperty("prompt_tokens").EnumerateArray().Select(id => id.GetInt32()).ToArray();
            var decode = root.GetProperty("decode_tokens_per_second");
            return new SweepRun(0, root.GetProperty("generated_text").GetString() ?? string.Empty, promptIds.Length,
                promptIds.SequenceEqual(prompt.Tokens), root.GetProperty("time_to_first_token_milliseconds").GetDouble(),
                root.GetProperty("total_generation_milliseconds").GetDouble(),
                root.GetProperty("generated_tokens").GetArrayLength(),
                decode.ValueKind == JsonValueKind.Number ? decode.GetDouble() : null,
                run.Wall, run.PeakFootprintBytes, run.PeakResidentBytes, null);
        }
        finally
        {
            File.Delete(tokensFile);
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    internal static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    internal static SweepRun Failed(ProcessRun run) => new(run.ExitCode, string.Empty, null, null, null, null, 0, null,
        run.Wall, run.PeakFootprintBytes, run.PeakResidentBytes, SynapseQualitySubject.Tail(run.Error));
}

/// <summary>
/// <c>llama-completion</c> with an explicit KV cache type; timings come from its <c>perf</c> lines. The prompt file
/// carries one extra newline because llama.cpp <c>-f</c> drops one.
/// </summary>
internal sealed partial class LlamaSweepSubject(SweepOptions options, string device, string kv) : ISweepSubject
{
    public string Name => $"llamacpp-{device}";

    public string KvCache => kv;

    public bool CpuOnly => device == "cpu";

    public async Task<SweepRun> RunAsync(QualityCase prompt, int context, CancellationToken cancellationToken)
    {
        var promptFile = Path.Combine(Path.GetTempPath(), $"synapse-sweep-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(promptFile, prompt.Prompt + "\n", cancellationToken).ConfigureAwait(false);
        try
        {
            List<string> arguments =
            [
                "-m", options.ModelPath, "-f", promptFile, "-c", SynapseSweepSubject.Number(context),
                "-n", SynapseSweepSubject.Number(options.MaxTokens), "--temp", "0", "-s", "0", "-no-cnv",
                "--no-display-prompt", "--no-escape", "--no-warmup", "-ctk", kv, "-ctv", kv,
                "-t", SynapseSweepSubject.Number(options.Threads),
            ];
            arguments.AddRange(device == "metal" ? ["-ngl", "99", "-fa", "on"] : ["-ngl", "0", "-dev", "none", "--no-op-offload"]);
            var run = await QualityProcess.RunAsync(options.LlamaCompletion, arguments, cancellationToken, sampleMemory: true)
                .ConfigureAwait(false);
            var promptTiming = Timing(PromptEval(), run.Error);
            var eval = Timing(Eval(), run.Error);
            if (run.ExitCode != 0 || promptTiming is null)
            {
                return SynapseSweepSubject.Failed(run);
            }

            var generated = (eval?.Count ?? 0) + 1;
            return new SweepRun(0, EndOfText().Replace(run.Output, string.Empty).Trim(), promptTiming.Value.Count,
                promptTiming.Value.Count == prompt.Tokens.Length, promptTiming.Value.Milliseconds,
                promptTiming.Value.Milliseconds + (eval?.Milliseconds ?? 0), generated,
                eval is { Count: > 0 } e ? e.Count / (e.Milliseconds / 1000) : null,
                run.Wall, run.PeakFootprintBytes, run.PeakResidentBytes, null);
        }
        finally
        {
            File.Delete(promptFile);
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private static (double Milliseconds, int Count)? Timing(Regex pattern, string text) =>
        pattern.Match(text) is { Success: true } match
            ? (double.Parse(match.Groups["ms"].Value, CultureInfo.InvariantCulture),
                int.Parse(match.Groups["count"].Value, CultureInfo.InvariantCulture))
            : null;

    [GeneratedRegex(@"prompt eval time =\s*(?<ms>[\d.]+) ms /\s*(?<count>\d+) tokens", RegexOptions.CultureInvariant)]
    private static partial Regex PromptEval();

    [GeneratedRegex(@"(?<!prompt )eval time =\s*(?<ms>[\d.]+) ms /\s*(?<count>\d+) runs", RegexOptions.CultureInvariant)]
    private static partial Regex Eval();

    [GeneratedRegex(@"\s*\[end of text\]\s*$", RegexOptions.CultureInvariant)]
    private static partial Regex EndOfText();
}
