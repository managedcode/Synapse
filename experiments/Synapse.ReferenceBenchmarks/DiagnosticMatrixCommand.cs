using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

internal static class DiagnosticMatrixCommand
{
    private static readonly string[] Subjects = ["synapse", "dotllm", "llamasharp", "llamacpp"];

    public static async Task<int> RunAsync(string[] arguments)
    {
        var options = DiagnosticMatrixArguments.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine("Usage: matrix --model <GGUF> --prompt <text> " +
                "--prompt-token-ids <ids> --expected-token-ids <ids> --expected-text <text> " +
                "--synapse-executable <path> --dotllm-executable <path> --dotllm-version <sha> " +
                "--llamacpp-executable <path> --llamacpp-version <sha> --output <new.json> " +
                "[--max-tokens 8] [--threads 8] [--warmups 3] [--measurements 5]");
            return 2;
        }

        try
        {
            if (File.Exists(options.OutputPath))
            {
                throw new IOException($"Matrix evidence already exists: {options.OutputPath}");
            }

            var evidence = await MeasureAsync(options).ConfigureAwait(false);
            await using var output = new FileStream(
                options.OutputPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            await JsonSerializer.SerializeAsync(
                output, evidence, BenchmarkJsonContext.Default.DiagnosticMatrixEvidence).ConfigureAwait(false);
            Console.WriteLine(Path.GetFullPath(options.OutputPath));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static async Task<DiagnosticMatrixEvidence> MeasureAsync(DiagnosticMatrixArguments options)
    {
        var samples = new List<DiagnosticMatrixSample>();
        for (var round = 0; round < options.Warmups + options.Measurements; round++)
        {
            for (var index = 0; index < Subjects.Length; index++)
            {
                var subject = Subjects[(round + index) % Subjects.Length];
                samples.Add(await MeasureSubjectAsync(options, subject, round,
                    round < options.Warmups).ConfigureAwait(false));
                Console.Error.WriteLine($"{subject} round {round + 1}/{options.Warmups + options.Measurements} complete");
            }
        }

        return new DiagnosticMatrixEvidence(
            samples.Any(sample => !sample.QualityMatched && !sample.Warmup)
                ? "ineligible_quality_mismatch"
                : "measured_diagnostic_no_statistical_verdict",
            await HashFileAsync(options.ModelPath).ConfigureAwait(false),
            await HashFileAsync(typeof(DiagnosticMatrixCommand).Assembly.Location).ConfigureAwait(false),
            await HashFileAsync(options.SynapseExecutable).ConfigureAwait(false),
            await HashFileAsync(options.DotLlmExecutable).ConfigureAwait(false),
            await HashFileAsync(options.LlamaCppExecutable).ConfigureAwait(false),
            await HashFileAsync(typeof(LLama.LLamaWeights).Assembly.Location).ConfigureAwait(false),
            RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            "not_recorded",
            options.Prompt,
            options.PromptTokenIds,
            options.ExpectedTokenIds,
            options.ExpectedText,
            options.Threads,
            options.MaxTokens,
            options.Warmups,
            options.Measurements,
            samples);
    }

    private static async Task<DiagnosticMatrixSample> MeasureSubjectAsync(
        DiagnosticMatrixArguments options, string subject, int round, bool warmup)
    {
        var wall = Stopwatch.StartNew();
        using var process = Process.Start(CreateStartInfo(options, subject))
            ?? throw new InvalidOperationException($"{subject} process did not start.");
        await using var sampler = new ProcessMemorySampler(process);
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        try
        {
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(5)).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync().ConfigureAwait(false);
            throw new TimeoutException($"{subject} exceeded the five-minute matrix timeout.");
        }

        wall.Stop();
        var outerMetrics = await sampler.CompleteAsync().ConfigureAwait(false);
        var stderr = await error.ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"{subject} exited {process.ExitCode}: {stderr}");
        }

        using var document = JsonDocument.Parse(await output.ConfigureAwait(false));
        var raw = document.RootElement.Clone();
        var qualityMatched = MatchesQuality(options, subject, raw);
        var inner = subject == "synapse" ? outerMetrics : ReadSubjectMetrics(raw);
        return new DiagnosticMatrixSample(
            subject, round, warmup, qualityMatched, raw,
            inner.MaximumObservedWorkingSetBytes,
            inner.PeakPhysicalFootprintBytes,
            inner.MemorySampleCount,
            subject == "synapse" ? "outer_direct_subject" : "inner_direct_subject",
            wall.Elapsed.TotalMilliseconds,
            outerMetrics.MaximumObservedWorkingSetBytes);
    }

    private static ProcessStartInfo CreateStartInfo(DiagnosticMatrixArguments options, string subject)
    {
        var start = new ProcessStartInfo
        {
            FileName = subject == "synapse" ? Path.GetFullPath(options.SynapseExecutable) : "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        var arguments = subject == "synapse" ? SynapseArguments(options) : ReferenceArguments(options, subject);
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        return start;
    }

    private static string[] SynapseArguments(DiagnosticMatrixArguments options) =>
    [
        "generate", "--model", options.ModelPath,
        "--tokens", string.Join(',', options.PromptTokenIds),
        "--max-tokens", options.MaxTokens.ToString(System.Globalization.CultureInfo.InvariantCulture),
        "--context-size", "512",
        "--threads", options.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
    ];

    private static string[] ReferenceArguments(DiagnosticMatrixArguments options, string subject)
    {
        var arguments = new List<string>
        {
            typeof(DiagnosticMatrixCommand).Assembly.Location,
            subject,
            "--model", options.ModelPath,
            "--prompt", options.Prompt,
            "--max-tokens", options.MaxTokens.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--threads", options.Threads.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--backend", "cpu",
        };
        if (subject is "dotllm" or "llamacpp")
        {
            arguments.Add("--subject-executable");
            arguments.Add(subject == "dotllm" ? options.DotLlmExecutable : options.LlamaCppExecutable);
            arguments.Add("--subject-version");
            arguments.Add(subject == "dotllm" ? options.DotLlmVersion : options.LlamaCppVersion);
        }

        if (subject == "llamacpp")
        {
            arguments.Add("--expected-prompt-token-ids");
            arguments.Add(string.Join(',', options.PromptTokenIds));
        }

        return [.. arguments];
    }

    private static bool MatchesQuality(DiagnosticMatrixArguments options, string subject, JsonElement result)
    {
        if (subject == "synapse")
        {
            var generated = result.GetProperty("generated_tokens").EnumerateArray()
                .Select(token => token.GetInt32());
            return generated.SequenceEqual(options.ExpectedTokenIds);
        }

        return result.GetProperty("generated_tokens").GetInt32() == options.MaxTokens &&
            result.GetProperty("text").GetString() == options.ExpectedText;
    }

    private static ObservedProcessMetrics ReadSubjectMetrics(JsonElement result) => new(
        TimeSpan.FromMilliseconds(result.GetProperty("process_cpu_milliseconds").GetDouble()),
        result.GetProperty("maximum_observed_working_set_bytes").GetInt64(),
        result.GetProperty("maximum_observed_private_virtual_bytes").ValueKind == JsonValueKind.Null
            ? null : result.GetProperty("maximum_observed_private_virtual_bytes").GetInt64(),
        result.GetProperty("maximum_observed_virtual_bytes").ValueKind == JsonValueKind.Null
            ? null : result.GetProperty("maximum_observed_virtual_bytes").GetInt64(),
        result.GetProperty("peak_physical_footprint_bytes").ValueKind == JsonValueKind.Null
            ? null : result.GetProperty("peak_physical_footprint_bytes").GetInt64(),
        result.GetProperty("memory_sample_count").GetInt32());

    private static async Task<string> HashFileAsync(string path)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream).ConfigureAwait(false));
    }
}

internal sealed record DiagnosticMatrixEvidence(
    string Status,
    string ModelSha256,
    string RunnerSha256,
    string SynapseBinarySha256,
    string DotLlmBinarySha256,
    string LlamaCppBinarySha256,
    string LlamaSharpAssemblySha256,
    string OperatingSystem,
    string Architecture,
    string PowerState,
    string Prompt,
    int[] PromptTokenIds,
    int[] ExpectedTokenIds,
    string ExpectedText,
    int Threads,
    int MaxTokens,
    int Warmups,
    int Measurements,
    IReadOnlyList<DiagnosticMatrixSample> Samples);

internal sealed record DiagnosticMatrixSample(
    string Subject,
    int Round,
    bool Warmup,
    bool QualityMatched,
    JsonElement SubjectResult,
    long? PeakResidentBytes,
    long? PeakPhysicalFootprintBytes,
    int MemorySampleCount,
    string MemoryMetricOrigin,
    double MatrixProcessWallMilliseconds,
    long? OuterProcessPeakResidentBytes);
