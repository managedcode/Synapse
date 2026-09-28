using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

return await ReferenceBenchmarkCommand.RunAsync(args).ConfigureAwait(false);

internal static class ReferenceBenchmarkCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var options = BenchmarkArguments.Parse(args);
        if (options is null)
        {
            PrintUsage();
            return 2;
        }

        try
        {
            var result = options.Subject switch
            {
                "dotllm" => await RunDotLlmAsync(options).ConfigureAwait(false),
                "llamasharp" => await RunLlamaSharpAsync(options).ConfigureAwait(false),
                "llamacpp" => await LlamaCppSubject.RunAsync(options).ConfigureAwait(false),
                _ => throw new UnreachableException(),
            };
            Console.WriteLine(JsonSerializer.Serialize(result, BenchmarkJsonContext.Default.BenchmarkResult));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static async Task<BenchmarkResult> RunLlamaSharpAsync(BenchmarkArguments options)
    {
        using var process = Process.GetCurrentProcess();
        var subjectTimer = Stopwatch.StartNew();
        var subjectCpuStart = process.TotalProcessorTime;
        _ = NativeLibraryConfig.All.WithLogCallback(static (_, _) => { });
        var parameters = new ModelParams(options.ModelPath)
        {
            BatchThreads = options.Threads,
            ContextSize = 512,
            GpuLayerCount = options.Backend == "metal" ? 99 : 0,
            Threads = options.Threads,
        };

        var loadTimer = Stopwatch.StartNew();
        using var model = await LLamaWeights.LoadFromFileAsync(parameters).ConfigureAwait(false);
        var executor = new StatelessExecutor(model, parameters);
        loadTimer.Stop();
        process.Refresh();
        var workingSetAfterLoad = process.WorkingSet64;

        var inference = new InferenceParams
        {
            MaxTokens = options.MaxTokens,
            SamplingPipeline = new GreedySamplingPipeline(),
        };

        var output = new System.Text.StringBuilder();
        var generationTimer = Stopwatch.StartNew();
        var firstTokenAt = (TimeSpan?)null;
        var generatedTokens = 0;
        await foreach (var piece in executor.InferAsync(options.Prompt, inference).ConfigureAwait(false))
        {
            firstTokenAt ??= generationTimer.Elapsed;
            generatedTokens++;
            _ = output.Append(piece);
        }

        generationTimer.Stop();
        subjectTimer.Stop();
        process.Refresh();
        var maximumObservedWorkingSet = Math.Max(workingSetAfterLoad, process.WorkingSet64);
        if (process.PeakWorkingSet64 > 0)
        {
            maximumObservedWorkingSet = Math.Max(maximumObservedWorkingSet, process.PeakWorkingSet64);
        }

        var subjectCpu = process.TotalProcessorTime - subjectCpuStart;
        var decodeSeconds = Math.Max(
            0.000_001,
            generationTimer.Elapsed.TotalSeconds - (firstTokenAt?.TotalSeconds ?? 0));

        return new BenchmarkResult(
            Subject: "llamasharp",
            SubjectVersion: options.SubjectVersion ?? GetAssemblyVersion(typeof(LLamaWeights).Assembly),
            Backend: options.Backend,
            ModelPath: Path.GetFullPath(options.ModelPath),
            Prompt: options.Prompt,
            MaxTokens: options.MaxTokens,
            Threads: options.Threads,
            GeneratedTokens: generatedTokens,
            Text: output.ToString(),
            LoadMilliseconds: loadTimer.Elapsed.TotalMilliseconds,
            TimeToFirstTokenMilliseconds: firstTokenAt?.TotalMilliseconds,
            TotalGenerationMilliseconds: generationTimer.Elapsed.TotalMilliseconds,
            DecodeTokensPerSecond: generatedTokens <= 1 ? null : (generatedTokens - 1) / decodeSeconds,
            SubjectWallMilliseconds: subjectTimer.Elapsed.TotalMilliseconds,
            ProcessCpuMilliseconds: subjectCpu.TotalMilliseconds,
            AverageCpuCores: AverageCpuCores(subjectCpu, subjectTimer.Elapsed),
            WorkingSetAfterLoadBytes: workingSetAfterLoad,
            MaximumObservedWorkingSetBytes: maximumObservedWorkingSet,
            MeasurementScope: "managed_wrapper_observed");
    }

    private static async Task<BenchmarkResult> RunDotLlmAsync(BenchmarkArguments options)
    {
        if (string.IsNullOrWhiteSpace(options.SubjectExecutable))
        {
            throw new ArgumentException("dotLLM requires --subject-executable <path>.");
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = Path.GetFullPath(options.SubjectExecutable),
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        AddDotLlmArguments(startInfo, options);

        var subjectTimer = Stopwatch.StartNew();
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("dotLLM process did not start.");
        var metricsTask = ObserveProcessAsync(process);
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(5)).ConfigureAwait(false);
        subjectTimer.Stop();
        var processMetrics = await metricsTask.ConfigureAwait(false);
        var output = await outputTask.ConfigureAwait(false);
        var error = await errorTask.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"dotLLM exited with {process.ExitCode}: {error}");
        }

        var result = JsonSerializer.Deserialize(output, BenchmarkJsonContext.Default.DotLlmResult)
            ?? throw new InvalidDataException("dotLLM returned an empty JSON result.");
        return new BenchmarkResult(
            Subject: "dotllm",
            SubjectVersion: options.SubjectVersion ?? "unknown",
            Backend: "cpu",
            ModelPath: Path.GetFullPath(options.ModelPath),
            Prompt: options.Prompt,
            MaxTokens: options.MaxTokens,
            Threads: options.Threads,
            GeneratedTokens: result.Usage.GeneratedTokens,
            Text: result.Text,
            LoadMilliseconds: result.Timings.LoadMilliseconds,
            TimeToFirstTokenMilliseconds: result.Timings.PrefillMilliseconds,
            TotalGenerationMilliseconds: result.Timings.TotalMilliseconds,
            DecodeTokensPerSecond: result.Timings.DecodeTokensPerSecond,
            SubjectWallMilliseconds: subjectTimer.Elapsed.TotalMilliseconds,
            ProcessCpuMilliseconds: processMetrics.Cpu.TotalMilliseconds,
            AverageCpuCores: AverageCpuCores(processMetrics.Cpu, subjectTimer.Elapsed),
            WorkingSetAfterLoadBytes: null,
            MaximumObservedWorkingSetBytes: processMetrics.MaximumObservedWorkingSetBytes,
            MeasurementScope: "dotllm_reported_plus_process_observed");
    }

    private static void AddDotLlmArguments(ProcessStartInfo startInfo, BenchmarkArguments options)
    {
        string[] arguments =
        [
            "run", options.ModelPath,
            "--prompt", options.Prompt,
            "--max-tokens", options.MaxTokens.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--temp", "0",
            "--threads", options.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--decode-threads", options.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--json",
        ];
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }
    }

    private static void PrintUsage() => Console.Error.WriteLine(
            "Usage: dotnet run --project experiments/Synapse.ReferenceBenchmarks -- " +
            "<dotllm|llamasharp|llamacpp> --model <path.gguf> --prompt <text> " +
            "[--max-tokens 32] [--threads 8] [--backend cpu|metal] " +
            "[--subject-executable <path>] [--subject-version <commit|version>] " +
            "[--expected-prompt-token-ids <comma-separated-ids>]");

    internal static double? AverageCpuCores(TimeSpan cpu, TimeSpan wall) => wall.TotalSeconds == 0
        ? null
        : cpu.TotalSeconds / wall.TotalSeconds;

    private static string GetAssemblyVersion(Assembly assembly) =>
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ??
        assembly.GetName().Version?.ToString() ??
        "unknown";

    internal static async Task<ObservedProcessMetrics> ObserveProcessAsync(Process process)
    {
        var cpu = TimeSpan.Zero;
        long? maximumObservedWorkingSetBytes = null;
        while (!process.HasExited)
        {
            try
            {
                process.Refresh();
                cpu = process.TotalProcessorTime;
                var observed = Math.Max(process.WorkingSet64, process.PeakWorkingSet64);
                if (observed > 0 &&
                    (maximumObservedWorkingSetBytes is null || observed > maximumObservedWorkingSetBytes))
                {
                    maximumObservedWorkingSetBytes = observed;
                }
            }
            catch (InvalidOperationException)
            {
                break;
            }

            await Task.Delay(TimeSpan.FromMilliseconds(5)).ConfigureAwait(false);
        }

        return new ObservedProcessMetrics(cpu, maximumObservedWorkingSetBytes);
    }
}

internal sealed record ObservedProcessMetrics(TimeSpan Cpu, long? MaximumObservedWorkingSetBytes);

internal sealed record BenchmarkArguments(
    string Subject,
    string ModelPath,
    string Prompt,
    int MaxTokens,
    int Threads,
    string Backend,
    string? SubjectExecutable,
    string? SubjectVersion,
    int[]? ExpectedPromptTokenIds)
{
    public static BenchmarkArguments? Parse(string[] args)
    {
        if (args.Length < 5 || args[0] is not ("dotllm" or "llamasharp" or "llamacpp"))
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 1; index < args.Length; index += 2)
        {
            if (index + 1 >= args.Length || !args[index].StartsWith("--", StringComparison.Ordinal))
            {
                return null;
            }

            values[args[index]] = args[index + 1];
        }

        if (!values.TryGetValue("--model", out var modelPath) ||
            !values.TryGetValue("--prompt", out var prompt))
        {
            return null;
        }

        var maxTokens = ReadPositiveInt(values, "--max-tokens", 32);
        var threads = ReadPositiveInt(values, "--threads", 8);
        var backend = values.GetValueOrDefault("--backend", "cpu");
        if (maxTokens <= 0 || threads <= 0 || backend is not ("cpu" or "metal"))
        {
            return null;
        }

        int[]? expectedPromptTokenIds = null;
        if (values.TryGetValue("--expected-prompt-token-ids", out var tokenIds))
        {
            var parts = tokenIds.Split(',', StringSplitOptions.TrimEntries);
            expectedPromptTokenIds = new int[parts.Length];
            for (var index = 0; index < parts.Length; index++)
            {
                if (!int.TryParse(parts[index], out expectedPromptTokenIds[index]) ||
                    expectedPromptTokenIds[index] < 0)
                {
                    return null;
                }
            }
        }

        return new BenchmarkArguments(
            args[0],
            modelPath,
            prompt,
            maxTokens,
            threads,
            backend,
            values.GetValueOrDefault("--subject-executable"),
            values.GetValueOrDefault("--subject-version"),
            expectedPromptTokenIds);
    }

    private static int ReadPositiveInt(Dictionary<string, string> values, string key, int fallback) => !values.TryGetValue(key, out var value)
            ? fallback
            : int.TryParse(value, out var parsed) ? parsed : -1;
}

internal sealed record BenchmarkResult(
    string Subject,
    string SubjectVersion,
    string Backend,
    string ModelPath,
    string Prompt,
    int MaxTokens,
    int Threads,
    int GeneratedTokens,
    string Text,
    double? LoadMilliseconds,
    double? TimeToFirstTokenMilliseconds,
    double? TotalGenerationMilliseconds,
    double? DecodeTokensPerSecond,
    double SubjectWallMilliseconds,
    double ProcessCpuMilliseconds,
    double? AverageCpuCores,
    long? WorkingSetAfterLoadBytes,
    long? MaximumObservedWorkingSetBytes,
    string MeasurementScope,
    int[]? PromptTokenIds = null,
    double? NativePromptEvalMilliseconds = null,
    double? NativeEvalMilliseconds = null,
    double? NativeEvalTokensPerSecond = null);

internal sealed record DotLlmResult(
    string Text,
    DotLlmUsage Usage,
    DotLlmTimings Timings);

internal sealed record DotLlmUsage(
    [property: System.Text.Json.Serialization.JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: System.Text.Json.Serialization.JsonPropertyName("generated_tokens")] int GeneratedTokens);

internal sealed record DotLlmTimings(
    [property: System.Text.Json.Serialization.JsonPropertyName("load_ms")] double LoadMilliseconds,
    [property: System.Text.Json.Serialization.JsonPropertyName("prefill_ms")] double PrefillMilliseconds,
    [property: System.Text.Json.Serialization.JsonPropertyName("total_ms")] double TotalMilliseconds,
    [property: System.Text.Json.Serialization.JsonPropertyName("decode_tok_s")] double DecodeTokensPerSecond);

[System.Text.Json.Serialization.JsonSerializable(typeof(BenchmarkResult))]
[System.Text.Json.Serialization.JsonSerializable(typeof(DotLlmResult))]
[System.Text.Json.Serialization.JsonSourceGenerationOptions(
    PropertyNamingPolicy = System.Text.Json.Serialization.JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class BenchmarkJsonContext : System.Text.Json.Serialization.JsonSerializerContext;
