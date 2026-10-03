using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Common;
using LLama.Native;

internal static class LockedDialogueCommand
{
    private static readonly string[] Subjects = ["synapse", "dotllm", "llamasharp", "llamacpp"];

    public static async Task<int> RunAsync(string[] arguments)
    {
        var options = DialogueOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine("Usage: dialogue --scenario <3-turn.json> --model <GGUF> " +
                "--synapse-executable <path> --dotllm-executable <path> --dotllm-version <sha> " +
                "--llamacpp-executable <path> --llamacpp-version <sha> --output <new.json> " +
                "[--max-tokens 64] [--threads 2] [--warmups 1] [--measurements 3]");
            return 2;
        }

        try
        {
            if (File.Exists(options.Output))
            {
                throw new IOException($"Dialogue evidence already exists: {options.Output}");
            }

            var turns = LockedChatScenario.Read(options.Scenario);
            var prompts = TokenizePrompts(options.Model, turns);
            var samples = new List<DialogueSample>();
            for (var round = 0; round < options.Warmups + options.Measurements; round++)
            {
                for (var turn = 0; turn < prompts.Count; turn++)
                {
                    for (var index = 0; index < Subjects.Length; index++)
                    {
                        var subject = Subjects[(round + turn + index) % Subjects.Length];
                        samples.Add(await MeasureAsync(options, prompts[turn], subject,
                            turn, round, round < options.Warmups).ConfigureAwait(false));
                        Console.Error.WriteLine($"{subject} turn {turn + 1}/{prompts.Count} round {round + 1}/" +
                            $"{options.Warmups + options.Measurements} complete");
                    }
                }
            }

            var evidence = new DialogueEvidence("measured_diagnostic_quality_unreviewed",
                await HashAsync(options.Scenario).ConfigureAwait(false),
                await HashAsync(options.Model).ConfigureAwait(false),
                await HashAsync(typeof(LockedDialogueCommand).Assembly.Location).ConfigureAwait(false),
                await HashAsync(options.Synapse).ConfigureAwait(false),
                await HashAsync(options.DotLlm).ConfigureAwait(false),
                await HashAsync(options.LlamaCpp).ConfigureAwait(false),
                await HashAsync(typeof(LLamaWeights).Assembly.Location).ConfigureAwait(false),
                options.DotLlmVersion, options.LlamaCppVersion,
                RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture.ToString(),
                options.Threads, options.MaxTokens, options.Warmups, options.Measurements,
                "locked_transcript_fresh_process_per_turn_no_kv_reuse", prompts, samples);
            await using var output = new FileStream(options.Output, FileMode.CreateNew,
                FileAccess.Write, FileShare.None);
            await JsonSerializer.SerializeAsync(output, evidence,
                BenchmarkJsonContext.Default.DialogueEvidence).ConfigureAwait(false);
            Console.WriteLine(Path.GetFullPath(options.Output));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static List<DialogueTurn> TokenizePrompts(string modelPath, List<DialogueTurn> turns)
    {
        _ = NativeLibraryConfig.All.WithLogCallback(static (_, _) => { });
        using var weights = LLamaWeights.LoadFromFile(new ModelParams(modelPath)
        {
            ContextSize = 512,
            GpuLayerCount = 0,
        });
        return [.. turns.Select(turn => turn with
        {
            PromptTokenIds = [.. weights.Tokenize(turn.Prompt, add_bos: false, special: true,
                    encoding: Encoding.UTF8)
                .Select(token => (int)token)],
        })];
    }

    private static async Task<DialogueSample> MeasureAsync(DialogueOptions options,
        DialogueTurn turn, string subject, int turnIndex, int round, bool warmup)
    {
        var start = new ProcessStartInfo
        {
            FileName = subject == "synapse" ? Path.GetFullPath(options.Synapse) : "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in SubjectArguments(options, turn, subject))
        {
            start.ArgumentList.Add(argument);
        }

        var timer = Stopwatch.StartNew();
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException($"{subject} did not start.");
        await using var sampler = new ProcessMemorySampler(process);
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        try
        {
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(5)).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync().ConfigureAwait(false);
            throw new TimeoutException($"{subject} turn {turnIndex + 1} exceeded five minutes.");
        }

        timer.Stop();
        var outer = await sampler.CompleteAsync().ConfigureAwait(false);
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"{subject} exited {process.ExitCode}: {await stderr}");
        }

        using var document = JsonDocument.Parse(await stdout.ConfigureAwait(false));
        var raw = document.RootElement.Clone();
        var generated = raw.GetProperty("generated_tokens");
        var outputCount = generated.ValueKind == JsonValueKind.Array
            ? generated.GetArrayLength() : generated.GetInt32();
        if (outputCount == 0)
        {
            throw new InvalidDataException($"{subject} returned no tokens at turn {turnIndex + 1}.");
        }

        var resident = subject == "synapse" ? outer.MaximumObservedWorkingSetBytes :
            raw.GetProperty("maximum_observed_working_set_bytes").GetInt64();
        var cpu = subject == "synapse" ? outer.Cpu.TotalMilliseconds :
            raw.GetProperty("process_cpu_milliseconds").GetDouble();
        return new DialogueSample(subject, turnIndex + 1, round, warmup,
            "unreviewed", outputCount, timer.Elapsed.TotalMilliseconds, cpu,
            resident, raw);
    }

    private static string[] SubjectArguments(DialogueOptions options, DialogueTurn turn, string subject)
    {
        if (subject == "synapse")
        {
            return ["generate", "--model", PreparedBenchmarkModel.Require(options.Model), "--tokens",
                string.Join(',', turn.PromptTokenIds), "--max-tokens", Number(options.MaxTokens),
                "--context-size", "512", "--threads", Number(options.Threads)];
        }

        var arguments = new List<string>
        {
            typeof(LockedDialogueCommand).Assembly.Location, subject,
            "--model", options.Model, "--prompt", turn.Prompt,
            "--max-tokens", Number(options.MaxTokens), "--threads", Number(options.Threads),
            "--backend", "cpu",
        };
        if (subject is "dotllm" or "llamacpp")
        {
            arguments.AddRange(["--subject-executable", subject == "dotllm" ? options.DotLlm : options.LlamaCpp,
                "--subject-version", subject == "dotllm" ? options.DotLlmVersion : options.LlamaCppVersion]);
        }

        if (subject == "llamacpp")
        {
            arguments.AddRange(["--expected-prompt-token-ids", string.Join(',', turn.PromptTokenIds)]);
        }

        return [.. arguments];
    }

    private static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static async Task<string> HashAsync(string path)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream).ConfigureAwait(false));
    }
}

internal sealed record DialogueSample(string Subject, int Turn, int Round, bool Warmup,
    string QualityStatus, int GeneratedTokens, double ProcessWallMilliseconds,
    double ProcessCpuMilliseconds, long? PeakResidentBytes, JsonElement SubjectResult);

internal sealed record DialogueEvidence(string Status, string ScenarioSha256, string ModelSha256,
    string RunnerSha256, string SynapseBinarySha256, string DotLlmBinarySha256,
    string LlamaCppBinarySha256, string LlamaSharpAssemblySha256,
    string DotLlmVersion, string LlamaCppVersion,
    string OperatingSystem, string Architecture, int Threads, int MaxTokens,
    int Warmups, int Measurements, string ContextMode,
    IReadOnlyList<DialogueTurn> Turns, IReadOnlyList<DialogueSample> Samples);

internal sealed record DialogueOptions(string Scenario, string Model, string Synapse,
    string DotLlm, string DotLlmVersion, string LlamaCpp, string LlamaCppVersion,
    string Output, int MaxTokens, int Threads, int Warmups, int Measurements)
{
    public static DialogueOptions? Parse(string[] args)
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
            return values.TryGetValue(name, out var value)
                ? int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var count) ? count : -1
                : fallback;
        }
        var options = new DialogueOptions(Get("--scenario"), Get("--model"),
            Get("--synapse-executable"), Get("--dotllm-executable"), Get("--dotllm-version"),
            Get("--llamacpp-executable"), Get("--llamacpp-version"), Get("--output"),
            Count("--max-tokens", 64), Count("--threads", 2),
            Count("--warmups", 1), Count("--measurements", 3));
        return File.Exists(options.Scenario) && File.Exists(options.Model) &&
            File.Exists(options.Synapse) && File.Exists(options.DotLlm) &&
            File.Exists(options.LlamaCpp) && !string.IsNullOrWhiteSpace(options.Output) &&
            !string.IsNullOrWhiteSpace(options.DotLlmVersion) &&
            !string.IsNullOrWhiteSpace(options.LlamaCppVersion) &&
            options.MaxTokens is > 0 and <= 128 && options.Threads is > 0 and <= 256 &&
            options.Warmups is >= 0 and <= 10 && options.Measurements is > 0 and <= 30
            ? options : null;
    }
}
