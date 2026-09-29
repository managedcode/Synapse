using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>An engine answering quality prompts; every engine runs as its own process (ADR-015).</summary>
internal interface IQualitySubject : IAsyncDisposable
{
    string Name { get; }

    Task<SubjectAnswer> AnswerAsync(QualityCase quality, CancellationToken cancellationToken);
}

/// <summary>One engine's answer with the token-identity evidence it could provide.</summary>
internal sealed record SubjectAnswer(
    string Output,
    int? PromptTokens,
    bool? TokenIdsIdentical,
    double? PromptMilliseconds,
    double WallMilliseconds,
    int ExitCode,
    string? Error);

/// <summary>
/// Runs <c>synapse generate --tokens-file</c> with the harness token IDs and decodes the output IDs. A variant adds a
/// layer drop (ADR-019): <c>drop-layers=3.5.7</c> (experimental) or <c>drop-profile=&lt;evidence.json&gt;</c> (qualified).
/// </summary>
internal sealed class SynapseQualitySubject(
    QualityRunOptions options,
    string backend,
    string kv,
    ITextTokenizer tokenizer,
    string? variant = null) : IQualitySubject
{
    private readonly string[] _variantArguments = variant?.Split('=', 2) switch
    {
        null => [],
        ["drop-layers", var layers] => ["--drop-layers", layers.Replace('.', ',')],
        ["drop-profile", var path] => ["--drop-profile", path],
        _ => throw new ArgumentException($"Unknown Synapse quality variant '{variant}'."),
    };

    public string Name => $"synapse-{backend}-kv{kv}" + (variant is null ? string.Empty : "-" + variant.Split('=')[0] +
        (variant.StartsWith("drop-layers=", StringComparison.Ordinal) ? variant["drop-layers=".Length..] : string.Empty));

    public async Task<SubjectAnswer> AnswerAsync(QualityCase quality, CancellationToken cancellationToken)
    {
        var tokensFile = Path.Combine(Path.GetTempPath(), $"synapse-quality-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(tokensFile, string.Join('\n', quality.Tokens), cancellationToken).ConfigureAwait(false);
        try
        {
            List<string> arguments =
            [
                "generate", "--model", options.ModelPath, "--tokens-file", tokensFile,
                "--max-tokens", Number(quality.MaxTokens), "--context-size", Number(options.ContextSize),
                "--backend", backend, "--kv-precision", kv, "--threads", Number(options.Threads),
            ];
            if (options.RopeScaling is { } scaling)
            {
                arguments.AddRange(["--rope-scaling", scaling]);
            }

            arguments.AddRange(_variantArguments);
            var run = await QualityProcess.RunAsync(options.SynapseExecutable, arguments, cancellationToken).ConfigureAwait(false);
            if (run.ExitCode != 0)
            {
                return new SubjectAnswer(string.Empty, null, null, null, run.Wall, run.ExitCode, Tail(run.Error));
            }

            using var json = JsonDocument.Parse(run.Output);
            var root = json.RootElement;
            var generated = root.GetProperty("generated_tokens").EnumerateArray().Select(id => id.GetInt32()).ToArray();
            var prompt = root.GetProperty("prompt_tokens").EnumerateArray().Select(id => id.GetInt32()).ToArray();
            return new SubjectAnswer(
                tokenizer.Decode(generated),
                prompt.Length,
                prompt.SequenceEqual(quality.Tokens),
                root.GetProperty("time_to_first_token_milliseconds").GetDouble(),
                run.Wall,
                0,
                null);
        }
        finally
        {
            File.Delete(tokensFile);
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    internal static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    internal static string Tail(string text) => text.Length <= 2_000 ? text : text[^2_000..];
}

/// <summary>
/// Runs <c>llama-completion</c> on the same prompt text with <c>--no-escape</c>. Token identity is checked with
/// <c>llama-tokenize</c> over the same file and the prompt count from the completion's own timing line.
/// </summary>
internal sealed partial class LlamaQualitySubject(QualityRunOptions options, string device) : IQualitySubject
{
    public string Name => $"llamacpp-{device}";

    public async Task<SubjectAnswer> AnswerAsync(QualityCase quality, CancellationToken cancellationToken)
    {
        // llama.cpp `-f` drops one trailing newline and llama-tokenize does not, so the files differ by exactly that.
        var promptFile = Path.Combine(Path.GetTempPath(), $"synapse-quality-{Guid.NewGuid():N}.txt");
        var completionFile = promptFile + ".completion";
        await File.WriteAllTextAsync(promptFile, quality.Prompt, cancellationToken).ConfigureAwait(false);
        await File.WriteAllTextAsync(completionFile, quality.Prompt + "\n", cancellationToken).ConfigureAwait(false);
        try
        {
            var identical = await SameTokensAsync(promptFile, quality.Tokens, cancellationToken).ConfigureAwait(false);
            var run = await QualityProcess.RunAsync(options.LlamaCompletion!, Arguments(completionFile, quality), cancellationToken)
                .ConfigureAwait(false);
            var timing = PromptTiming().Match(run.Error);
            int? count = timing.Success ? int.Parse(timing.Groups["count"].Value, CultureInfo.InvariantCulture) : null;
            return new SubjectAnswer(
                EndOfText().Replace(run.Output, string.Empty).Trim(),
                count,
                identical && count == quality.Tokens.Length,
                timing.Success ? double.Parse(timing.Groups["ms"].Value, CultureInfo.InvariantCulture) : null,
                run.Wall,
                run.ExitCode,
                run.ExitCode == 0 ? null : SynapseQualitySubject.Tail(run.Error));
        }
        finally
        {
            File.Delete(promptFile);
            File.Delete(completionFile);
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private List<string> Arguments(string promptFile, QualityCase quality)
    {
        List<string> arguments =
        [
            "-m", options.ModelPath, "-f", promptFile, "-c", SynapseQualitySubject.Number(options.ContextSize),
            "-n", SynapseQualitySubject.Number(quality.MaxTokens), "--temp", "0", "-s", "0", "-no-cnv",
            "--no-display-prompt", "--no-escape", "--no-warmup", "-t", SynapseQualitySubject.Number(options.Threads),
        ];
        arguments.AddRange(device == "metal" ? ["-ngl", "99", "-fa", "on"] : ["-ngl", "0", "-dev", "none", "--no-op-offload"]);
        if (options.RopeScaling is { } scaling)
        {
            var parts = scaling.Split(':');
            arguments.AddRange(["--rope-scaling", "yarn", "--rope-scale", parts[1], "--yarn-orig-ctx", parts[2]]);
        }

        return arguments;
    }

    private async Task<bool> SameTokensAsync(string promptFile, int[] expected, CancellationToken cancellationToken)
    {
        var run = await QualityProcess.RunAsync(
            options.LlamaTokenize!,
            ["-m", options.ModelPath, "-f", promptFile, "--ids", "--no-escape", "--log-disable"],
            cancellationToken).ConfigureAwait(false);
        var ids = run.Output.Trim().Trim('[', ']')
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(value => int.Parse(value, CultureInfo.InvariantCulture));
        return run.ExitCode == 0 && ids.SequenceEqual(expected);
    }

    [GeneratedRegex(@"\s*\[end of text\]\s*$", RegexOptions.CultureInvariant)]
    private static partial Regex EndOfText();

    [GeneratedRegex(@"prompt eval time =\s*(?<ms>[\d.]+) ms /\s*(?<count>\d+) tokens", RegexOptions.CultureInvariant)]
    private static partial Regex PromptTiming();
}

/// <summary>Captured output of one finished child process, with sampled peaks when requested.</summary>
internal sealed record ProcessRun(
    int ExitCode,
    string Output,
    string Error,
    double Wall,
    long? PeakFootprintBytes = null,
    long? PeakResidentBytes = null);

internal static class QualityProcess
{
    public static async Task<ProcessRun> RunAsync(
        string executable,
        IEnumerable<string> arguments,
        CancellationToken cancellationToken,
        bool sampleMemory = false)
    {
        var start = new ProcessStartInfo
        {
            FileName = executable,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        var timer = Stopwatch.StartNew();
        using var process = Process.Start(start) ?? throw new InvalidOperationException($"{executable} did not start.");
        var output = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var error = process.StandardError.ReadToEndAsync(cancellationToken);
        ObservedProcessMetrics? metrics = null;
        if (sampleMemory)
        {
            await using var sampler = new ProcessMemorySampler(process);
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            metrics = await sampler.CompleteAsync().ConfigureAwait(false);
        }
        else
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        }

        return new ProcessRun(process.ExitCode, await output.ConfigureAwait(false), await error.ConfigureAwait(false),
            timer.Elapsed.TotalMilliseconds, metrics?.PeakPhysicalFootprintBytes, metrics?.MaximumObservedWorkingSetBytes);
    }
}
