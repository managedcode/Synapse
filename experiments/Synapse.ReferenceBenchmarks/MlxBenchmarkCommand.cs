using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

internal static partial class MlxBenchmarkCommand
{
    public static async Task<int> RunAsync(string[] arguments)
    {
        var options = MlxOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine("Usage: mlx --binary <SwiftLM> --model <local-MLX-directory> " +
                "--scenario <1-or-3-turn.json> --output <new.json> --binary-version <revision> " +
                "[--port 15414] [--max-tokens 64] [--warmups 1] [--measurements 3]");
            return 2;
        }

        try
        {
            if (File.Exists(options.Output))
            {
                throw new IOException($"MLX evidence already exists: {options.Output}");
            }

            var turns = LockedChatScenario.Read(options.Scenario);
            var samples = await MeasureAsync(options, turns).ConfigureAwait(false);
            var evidence = new MlxEvidence("measured_metal_separate_weight_cohort_quality_unreviewed",
                options.BinaryVersion,
                await HashAsync(options.Binary).ConfigureAwait(false),
                await HashAsync(Path.Combine(Path.GetDirectoryName(options.Binary)!, "mlx.metallib"))
                    .ConfigureAwait(false),
                await HashAsync(Path.Combine(options.Model, "model.safetensors"))
                    .ConfigureAwait(false),
                await HashAsync(options.Scenario).ConfigureAwait(false),
                options.Model, options.MaxTokens, options.Warmups, options.Measurements,
                "resident_server_locked_transcript_cache_hit_observed_from_server_log",
                turns, samples);
            await using var output = new FileStream(options.Output, FileMode.CreateNew,
                FileAccess.Write, FileShare.None);
            await JsonSerializer.SerializeAsync(output, evidence,
                BenchmarkJsonContext.Default.MlxEvidence).ConfigureAwait(false);
            Console.WriteLine(Path.GetFullPath(options.Output));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static async Task<List<MlxSample>> MeasureAsync(MlxOptions options,
        List<DialogueTurn> turns)
    {
        var start = new ProcessStartInfo(options.Binary)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in new[] { "--model", options.Model, "--host", "127.0.0.1",
            "--port", Number(options.Port), "--ctx-size", "512", "--temp", "0",
            "--max-tokens", Number(options.MaxTokens) })
        {
            start.ArgumentList.Add(argument);
        }

        var load = Stopwatch.StartNew();
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("SwiftLM process did not start.");
        var logs = new ConcurrentQueue<string>();
        var ready = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var stdout = PumpAsync(process.StandardOutput, logs, ready);
        var stderr = PumpAsync(process.StandardError, logs, ready);
        await using var sampler = new ProcessMemorySampler(process);
        try
        {
            await ready.Task.WaitAsync(TimeSpan.FromMinutes(2)).ConfigureAwait(false);
            load.Stop();
            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
            var samples = new List<MlxSample>();
            for (var round = 0; round < options.Warmups + options.Measurements; round++)
            {
                for (var index = 0; index < turns.Count; index++)
                {
                    var before = logs.Count;
                    process.Refresh();
                    var cpuBefore = process.TotalProcessorTime;
                    var response = await RequestAsync(client, options, turns, index)
                        .ConfigureAwait(false);
                    process.Refresh();
                    var cpu = (process.TotalProcessorTime - cpuBefore).TotalMilliseconds;
                    var relevantLogs = logs.ToArray().Skip(before).ToArray();
                    var hit = relevantLogs.Select(line => CacheHit().Match(line))
                        .FirstOrDefault(match => match.Success);
                    var resident = Math.Max(process.WorkingSet64, process.PeakWorkingSet64);
                    samples.Add(new MlxSample(index + 1, round, round < options.Warmups,
                        "unreviewed", response.PromptTokens, response.GeneratedTokens,
                        response.Text, load.Elapsed.TotalMilliseconds,
                        response.TimeToFirstTokenMilliseconds, response.WallMilliseconds,
                        response.DecodeTokensPerSecond, cpu, resident > 0 ? resident : null,
                        MacProcessFootprint.TryReadBytes(process.Id),
                        hit?.Success == true ? int.Parse(hit.Groups[1].Value, CultureInfo.InvariantCulture) : null,
                        hit?.Success == true ? "observed_hit" : "not_observed",
                        response.RawUsage));
                    Console.Error.WriteLine($"mlx turn {index + 1}/{turns.Count} round " +
                        $"{round + 1}/{options.Warmups + options.Measurements} complete");
                }
            }

            _ = await sampler.CompleteAsync().ConfigureAwait(false);
            return samples;
        }
        finally
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            await process.WaitForExitAsync().ConfigureAwait(false);
            await Task.WhenAll(stdout, stderr).ConfigureAwait(false);
        }
    }

    private static async Task PumpAsync(StreamReader reader, ConcurrentQueue<string> logs,
        TaskCompletionSource ready)
    {
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            logs.Enqueue(line);
            if (line.Contains("Ready. Listening", StringComparison.Ordinal) ||
                line.Contains("\"event\":\"ready\"", StringComparison.Ordinal))
            {
                _ = ready.TrySetResult();
            }
        }

        _ = ready.TrySetException(new InvalidOperationException(
            "SwiftLM exited before reporting readiness; inspect the external binary and model."));
    }

    private static async Task<MlxResponse> RequestAsync(HttpClient client, MlxOptions options,
        List<DialogueTurn> turns, int index)
    {
        var messages = new List<MlxMessage> { new("system", turns[0].System) };
        for (var previous = 0; previous < index; previous++)
        {
            messages.Add(new MlxMessage("user", turns[previous].User));
            messages.Add(new MlxMessage("assistant", turns[previous].LockedAssistant));
        }

        messages.Add(new MlxMessage("user", turns[index].User));
        var request = new MlxRequest(options.Model, messages, options.MaxTokens,
            0, true, new MlxStreamOptions(true));
        using var message = new HttpRequestMessage(HttpMethod.Post,
            $"http://127.0.0.1:{options.Port}/v1/chat/completions")
        {
            Content = JsonContent.Create(request, BenchmarkJsonContext.Default.MlxRequest),
        };
        var timer = Stopwatch.StartNew();
        using var response = await client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead)
            .ConfigureAwait(false);
        _ = response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
        using var reader = new StreamReader(stream);
        var text = new System.Text.StringBuilder();
        double? first = null;
        JsonElement? usage = null;
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal) || line == "data: [DONE]")
            {
                continue;
            }

            using var chunk = JsonDocument.Parse(line[6..]);
            var root = chunk.RootElement;
            if (root.TryGetProperty("usage", out var currentUsage))
            {
                usage = currentUsage.Clone();
            }

            if (root.GetProperty("choices").GetArrayLength() > 0 &&
                root.GetProperty("choices")[0].GetProperty("delta")
                    .TryGetProperty("content", out var content) &&
                content.ValueKind == JsonValueKind.String && content.GetString() is { Length: > 0 } piece)
            {
                first ??= timer.Elapsed.TotalMilliseconds;
                _ = text.Append(piece);
            }
        }

        timer.Stop();
        if (usage is null || first is null)
        {
            throw new InvalidDataException("SwiftLM streaming response omitted usage or first text token.");
        }

        var promptTokens = usage.Value.GetProperty("prompt_tokens").GetInt32();
        var generatedTokens = usage.Value.GetProperty("completion_tokens").GetInt32();
        var decodeSeconds = Math.Max(0.000_001, (timer.Elapsed.TotalMilliseconds - first.Value) / 1000d);
        return new MlxResponse(promptTokens, generatedTokens, text.ToString(), first.Value,
            timer.Elapsed.TotalMilliseconds, generatedTokens > 1
                ? (generatedTokens - 1) / decodeSeconds : null, usage.Value);
    }

    private static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static async Task<string> HashAsync(string path)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream).ConfigureAwait(false));
    }

    [GeneratedRegex("Prompt cache HIT: ([0-9]+)/[0-9]+ tokens reused", RegexOptions.CultureInvariant)]
    private static partial Regex CacheHit();
}

internal sealed record MlxMessage(string Role, string Content);
internal sealed record MlxStreamOptions(bool IncludeUsage);
internal sealed record MlxRequest(string Model, List<MlxMessage> Messages, int MaxTokens,
    int Temperature, bool Stream, MlxStreamOptions StreamOptions);
internal sealed record MlxResponse(int PromptTokens, int GeneratedTokens, string Text,
    double TimeToFirstTokenMilliseconds, double WallMilliseconds,
    double? DecodeTokensPerSecond, JsonElement RawUsage);
internal sealed record MlxSample(int Turn, int Round, bool Warmup, string QualityStatus,
    int PromptTokens, int GeneratedTokens, string Text, double LoadMilliseconds,
    double TimeToFirstTokenMilliseconds, double RequestWallMilliseconds,
    double? DecodeTokensPerSecond, double ProcessCpuMilliseconds,
    long? PeakResidentBytes, long? PhysicalFootprintBytes, int? CacheHitTokens,
    string CacheStatus, JsonElement RawUsage);
internal sealed record MlxEvidence(string Status, string BinaryVersion, string BinarySha256,
    string MetalLibrarySha256, string ModelWeightsSha256, string ScenarioSha256,
    string ModelPath, int MaxTokens, int Warmups, int Measurements, string ContextMode,
    IReadOnlyList<DialogueTurn> Turns, IReadOnlyList<MlxSample> Samples);

internal sealed record MlxOptions(string Binary, string Model, string Scenario,
    string Output, string BinaryVersion, int Port, int MaxTokens, int Warmups, int Measurements)
{
    public static MlxOptions? Parse(string[] args)
    {
        if (args.Length == 0 || args.Length % 2 != 0)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (!args[index].StartsWith("--", StringComparison.Ordinal) ||
                !values.TryAdd(args[index], args[index + 1]))
            {
                return null;
            }
        }

        string Get(string name)
        {
            return values.GetValueOrDefault(name) ?? string.Empty;
        }

        int Count(string name, int fallback)
        {
            return values.TryGetValue(name, out var value) &&
                int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var count)
                ? count : values.ContainsKey(name) ? -1 : fallback;
        }

        var options = new MlxOptions(Get("--binary"), Get("--model"), Get("--scenario"),
            Get("--output"), Get("--binary-version"), Count("--port", 15414),
            Count("--max-tokens", 64), Count("--warmups", 1), Count("--measurements", 3));
        return File.Exists(options.Binary) && File.Exists(Path.Combine(options.Model, "model.safetensors")) &&
            File.Exists(Path.Combine(Path.GetDirectoryName(options.Binary)!, "mlx.metallib")) &&
            File.Exists(options.Scenario) && !string.IsNullOrWhiteSpace(options.Output) &&
            !string.IsNullOrWhiteSpace(options.BinaryVersion) && options.Port is > 1024 and <= 65535 &&
            options.MaxTokens is > 0 and <= 128 && options.Warmups is >= 0 and <= 10 &&
            options.Measurements is > 0 and <= 30 ? options : null;
    }
}
