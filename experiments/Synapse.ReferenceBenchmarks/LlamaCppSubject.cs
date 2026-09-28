using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;

internal static class LlamaCppSubject
{
    public static async Task<BenchmarkResult> RunAsync(BenchmarkArguments options)
    {
        ValidateOptions(options);
        var capture = await ExecuteAsync(options).ConfigureAwait(false);
        var promptTokenIds = ReadPromptTokenIds(capture.Diagnostics);
        if (options.ExpectedPromptTokenIds is { } expected &&
            !promptTokenIds.SequenceEqual(expected))
        {
            throw new InvalidDataException(
                $"llama.cpp prompt token IDs differ: expected [{string.Join(',', expected)}], " +
                $"actual [{string.Join(',', promptTokenIds)}].");
        }

        return CreateResult(options, capture, promptTokenIds);
    }

    private static void ValidateOptions(BenchmarkArguments options)
    {
        if (options.Backend != "cpu")
        {
            throw new ArgumentException("The direct llama.cpp baseline currently supports CPU only.");
        }

        if (string.IsNullOrWhiteSpace(options.SubjectExecutable) ||
            !File.Exists(options.SubjectExecutable))
        {
            throw new ArgumentException("llama.cpp requires --subject-executable <llama-completion>.");
        }

        if (string.IsNullOrWhiteSpace(options.SubjectVersion))
        {
            throw new ArgumentException("llama.cpp requires --subject-version <pinned revision>.");
        }
    }

    private static async Task<CompletionCapture> ExecuteAsync(BenchmarkArguments options)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = Path.GetFullPath(options.SubjectExecutable!),
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        AddArguments(startInfo, options);

        var subjectTimer = Stopwatch.StartNew();
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("llama-completion process did not start.");
        var metricsTask = ReferenceBenchmarkCommand.ObserveProcessAsync(process);
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            await process.WaitForExitAsync().ConfigureAwait(false);
            throw new TimeoutException("llama-completion exceeded the five-minute subject timeout.");
        }

        subjectTimer.Stop();
        var metrics = await metricsTask.ConfigureAwait(false);
        var text = (await outputTask.ConfigureAwait(false)).TrimEnd('\r', '\n');
        var diagnostics = await errorTask.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"llama-completion exited with {process.ExitCode}: {diagnostics}");
        }

        return new CompletionCapture(text, diagnostics, metrics, subjectTimer.Elapsed);
    }

    private static BenchmarkResult CreateResult(
        BenchmarkArguments options,
        CompletionCapture capture,
        int[] promptTokenIds)
    {
        var (promptEvalMs, promptTokens) = ReadTiming(capture.Diagnostics, "prompt eval time", "tokens");
        var (evalMs, evalRuns) = ReadTiming(capture.Diagnostics, "eval time", "runs");
        var generatedTokens = evalMs == 0 && evalRuns == 1 ? 1 : evalRuns + 1;
        if (promptTokens != promptTokenIds.Length || generatedTokens > options.MaxTokens)
        {
            throw new InvalidDataException("llama.cpp reported inconsistent prompt/decode token counts.");
        }

        return new BenchmarkResult(
            Subject: "llamacpp",
            SubjectVersion: options.SubjectVersion!,
            Backend: "cpu",
            ModelPath: Path.GetFullPath(options.ModelPath),
            Prompt: options.Prompt,
            MaxTokens: options.MaxTokens,
            Threads: options.Threads,
            GeneratedTokens: generatedTokens,
            Text: capture.Text,
            LoadMilliseconds: null,
            TimeToFirstTokenMilliseconds: null,
            TotalGenerationMilliseconds: null,
            DecodeTokensPerSecond: null,
            SubjectWallMilliseconds: capture.Wall.TotalMilliseconds,
            ProcessCpuMilliseconds: capture.Metrics.Cpu.TotalMilliseconds,
            AverageCpuCores: ReferenceBenchmarkCommand.AverageCpuCores(capture.Metrics.Cpu, capture.Wall),
            WorkingSetAfterLoadBytes: null,
            MaximumObservedWorkingSetBytes: capture.Metrics.MaximumObservedWorkingSetBytes,
            MeasurementScope: "native_internal_plus_process_observed",
            PromptTokenIds: promptTokenIds,
            NativePromptEvalMilliseconds: promptEvalMs,
            NativeEvalMilliseconds: evalMs,
            NativeEvalTokensPerSecond: evalMs <= 0 || generatedTokens <= 1
                ? null
                : evalRuns / (evalMs / 1000));
    }

    private sealed record CompletionCapture(
        string Text,
        string Diagnostics,
        ObservedProcessMetrics Metrics,
        TimeSpan Wall);

    private static void AddArguments(ProcessStartInfo startInfo, BenchmarkArguments options)
    {
        string[] arguments =
        [
            "--model", options.ModelPath,
            "--prompt", options.Prompt,
            "--n-predict", options.MaxTokens.ToString(CultureInfo.InvariantCulture),
            "--ctx-size", "512",
            "--threads", options.Threads.ToString(CultureInfo.InvariantCulture),
            "--threads-batch", options.Threads.ToString(CultureInfo.InvariantCulture),
            "--n-gpu-layers", "0",
            "--device", "none",
            "--fit", "off",
            "--no-kv-offload",
            "--temp", "0",
            "--seed", "1",
            "--no-warmup",
            "--no-display-prompt",
            "--verbose-prompt",
            "--perf",
            "--simple-io",
            "--no-conversation",
            "--color", "off",
        ];
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }
    }

    private static int[] ReadPromptTokenIds(string diagnostics)
    {
        var countMatch = Regex.Match(
            diagnostics,
            @"number of tokens in prompt =\s*(?<count>\d+)",
            RegexOptions.CultureInvariant,
            TimeSpan.FromSeconds(1));
        if (!countMatch.Success)
        {
            throw new InvalidDataException("llama.cpp did not report the prompt token count.");
        }

        var count = int.Parse(countMatch.Groups["count"].Value, CultureInfo.InvariantCulture);
        var matches = Regex.Matches(
            diagnostics,
            @"(?m)^\S+\s+I\s+(?<id>\d+)\s+->\s+'",
            RegexOptions.CultureInvariant,
            TimeSpan.FromSeconds(1));
        var ids = new int[matches.Count];
        for (var index = 0; index < matches.Count; index++)
        {
            ids[index] = int.Parse(matches[index].Groups["id"].Value, CultureInfo.InvariantCulture);
        }

        if (ids.Length != count || count == 0)
        {
            throw new InvalidDataException("llama.cpp verbose prompt IDs are missing or inconsistent.");
        }

        return ids;
    }

    private static (double Milliseconds, int Count) ReadTiming(
        string diagnostics,
        string label,
        string countUnit)
    {
        var pattern = @"common_perf_print:\s+" + Regex.Escape(label) +
            @"\s*=\s*(?<ms>[\d.]+)\s*ms\s*/\s*(?<count>\d+)\s*" + countUnit;
        var match = Regex.Match(
            diagnostics,
            pattern,
            RegexOptions.CultureInvariant,
            TimeSpan.FromSeconds(1));
        if (!match.Success)
        {
            throw new InvalidDataException($"llama.cpp did not report {label}.");
        }

        return (
            double.Parse(match.Groups["ms"].Value, CultureInfo.InvariantCulture),
            int.Parse(match.Groups["count"].Value, CultureInfo.InvariantCulture));
    }
}
