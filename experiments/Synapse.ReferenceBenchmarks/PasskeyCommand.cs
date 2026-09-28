using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;

/// <summary>
/// Long-context retrieval diagnostic: builds pass-key prompts from the locked token fixture at several
/// lengths and depths, runs the Synapse CLI as a fresh process per prompt, and writes raw JSON with the
/// answer check, timings, and whole-process memory. It is a diagnostic, not a paired benchmark (ADR-005).
/// </summary>
internal static class PasskeyCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var options = PasskeyArguments.Parse(args);
        if (options is null)
        {
            await Console.Error.WriteLineAsync(
                "Usage: passkey --synapse-executable <path> --model <gguf> --scenario <passkey.json> " +
                "--prompt-tokens <n,n,...> --depths <0.1,0.5,...> --backend <name> --context-size <n> " +
                "--output <file.json> [--rope-scaling yarn:<factor>:<trained>] [--threads <n>] [--max-tokens <n>]")
                .ConfigureAwait(false);
            return 2;
        }

        var scenario = JsonSerializer.Deserialize(
            await File.ReadAllTextAsync(options.Scenario).ConfigureAwait(false),
            PasskeyJsonContext.Default.PasskeyScenario) ?? throw new InvalidDataException("Empty scenario.");
        var runs = new List<PasskeyRun>();
        var index = 0;
        foreach (var promptTokens in options.PromptTokens)
        {
            foreach (var depth in options.Depths)
            {
                var key = scenario.Keys[index++ % scenario.Keys.Count];
                var prompt = PasskeyPrompt.Build(scenario.Segments, key.Needle, promptTokens, depth);
                runs.Add(await RunOnceAsync(options, prompt, key, depth).ConfigureAwait(false));
                var last = runs[^1];
                await Console.Error.WriteLineAsync(string.Create(
                    CultureInfo.InvariantCulture,
                    $"passkey prompt={last.PromptTokens} depth={depth} found={last.ContainsAnswer} ttft_ms={last.TimeToFirstTokenMilliseconds:F0}"))
                    .ConfigureAwait(false);
            }
        }

        var evidence = new PasskeyEvidence(
            1,
            "passkey-long-context-diagnostic",
            DateTimeOffset.UtcNow,
            RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            Environment.ProcessorCount,
            scenario.Id,
            Path.GetFullPath(options.ModelPath),
            options.Backend,
            options.ContextSize,
            options.RopeScaling,
            options.Threads,
            options.KvPrecision,
            runs);
        await File.WriteAllTextAsync(
            options.Output,
            JsonSerializer.Serialize(evidence, PasskeyJsonContext.Default.PasskeyEvidence)).ConfigureAwait(false);
        return runs.All(run => run.ExitCode == 0) ? 0 : 1;
    }

    private static async Task<PasskeyRun> RunOnceAsync(PasskeyArguments options, int[] prompt, PasskeyKey key, double depth)
    {
        var tokensFile = Path.Combine(Path.GetTempPath(), $"synapse-passkey-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(tokensFile, string.Join('\n', prompt)).ConfigureAwait(false);
        try
        {
            var start = new ProcessStartInfo
            {
                FileName = Path.GetFullPath(options.SynapseExecutable),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            foreach (var argument in SynapseArguments(options, tokensFile))
            {
                start.ArgumentList.Add(argument);
            }

            var wall = Stopwatch.StartNew();
            using var process = Process.Start(start) ?? throw new InvalidOperationException("Synapse did not start.");
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            ObservedProcessMetrics metrics;
            await using (var sampler = new ProcessMemorySampler(process))
            {
                await process.WaitForExitAsync().ConfigureAwait(false);
                metrics = await sampler.CompleteAsync().ConfigureAwait(false);
            }

            wall.Stop();
            return PasskeyRun.From(prompt.Length, depth, key, process.ExitCode, await output.ConfigureAwait(false),
                await error.ConfigureAwait(false), metrics, wall.Elapsed);
        }
        finally
        {
            File.Delete(tokensFile);
        }
    }

    private static List<string> SynapseArguments(PasskeyArguments options, string tokensFile)
    {
        var arguments = new List<string>
        {
            "generate", "--model", options.ModelPath, "--tokens-file", tokensFile,
            "--max-tokens", options.MaxTokens.ToString(CultureInfo.InvariantCulture),
            "--context-size", options.ContextSize.ToString(CultureInfo.InvariantCulture),
            "--threads", options.Threads.ToString(CultureInfo.InvariantCulture),
            "--backend", options.Backend,
        };
        if (options.RopeScaling is { } scaling)
        {
            arguments.AddRange(["--rope-scaling", scaling]);
        }

        arguments.AddRange(["--kv-precision", options.KvPrecision]);

        return arguments;
    }
}

/// <summary>Prompt = prefix, filler repeated to the target length with the needle at the requested depth, suffix.</summary>
internal static class PasskeyPrompt
{
    public static int[] Build(PasskeySegments segments, int[] needle, int targetTokens, double depth)
    {
        var fixedTokens = segments.Prefix.Length + needle.Length + segments.Suffix.Length;
        var fillers = Math.Max(1, (targetTokens - fixedTokens) / segments.Filler.Length);
        var needleAt = (int)Math.Round(Math.Clamp(depth, 0, 1) * fillers);
        var prompt = new List<int>(fixedTokens + (fillers * segments.Filler.Length));
        prompt.AddRange(segments.Prefix);
        for (var filler = 0; filler < fillers; filler++)
        {
            if (filler == needleAt)
            {
                prompt.AddRange(needle);
            }

            prompt.AddRange(segments.Filler);
        }

        if (needleAt == fillers)
        {
            prompt.AddRange(needle);
        }

        prompt.AddRange(segments.Suffix);
        return [.. prompt];
    }
}
