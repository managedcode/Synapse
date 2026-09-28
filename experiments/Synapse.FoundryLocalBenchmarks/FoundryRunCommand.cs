using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.AI.Foundry.Local;

internal static class FoundryRunCommand
{
    public static async Task<int> RunAsync(FoundryArguments arguments, CancellationToken cancellationToken)
    {
        var options = FoundryRunOptions.From(arguments);
        if (File.Exists(options.Output))
        {
            throw new IOException($"Foundry Local evidence already exists: {options.Output}");
        }

        var set = FoundryModelSet.Load(options.Set);
        var model = set.GetModel(options.Alias);
        var turns = LockedChatScenario.Read(options.Scenario);
        using var process = Process.GetCurrentProcess();
        await using var sampler = new ProcessMemorySampler(process);
        var prepared = await PrepareAsync(options, model.Variant(options.Device), set.ContextTokens,
            cancellationToken).ConfigureAwait(false);
        using var genai = prepared.GenAiConfig;
        var load = Stopwatch.StartNew();
        await prepared.Variant.LoadAsync(cancellationToken).ConfigureAwait(false);
        var loadMilliseconds = load.Elapsed.TotalMilliseconds;
        var samples = await MeasureAsync(prepared.Variant, turns, options, model.SystemPromptMode, process,
            cancellationToken).ConfigureAwait(false);
        await prepared.Variant.UnloadAsync(cancellationToken).ConfigureAwait(false);
        var metrics = await sampler.CompleteAsync().ConfigureAwait(false);
        var evidence = new FoundryEvidence(
            "measured_foundry_local_separate_onnx_cohort_quality_unreviewed", "foundry-local",
            "Microsoft.AI.Foundry.Local", FoundryRuntime.PackageVersion("Microsoft.AI.Foundry.Local"),
            FoundryRuntime.SdkAssemblyVersion(), FoundryRuntime.PackageVersion("Microsoft.ML.OnnxRuntime"),
            FoundryRuntime.PackageVersion("Microsoft.ML.OnnxRuntimeGenAI.Foundry"),
            await HashAsync(FoundryRuntime.NativeRuntimePath(), cancellationToken).ConfigureAwait(false),
            set.Id, await HashAsync(options.Set, cancellationToken).ConfigureAwait(false),
            await HashAsync(options.Scenario, cancellationToken).ConfigureAwait(false),
            model.Family, model.Alias, prepared.Variant.Id, options.Device,
            prepared.Variant.Info.Runtime?.ExecutionProvider ?? "unknown", prepared.Variant.Info.FileSizeMb ?? 0,
            prepared.Variant.Info.License ?? "unknown", GenAiModelType(genai), GenAiSearch(genai),
            prepared.Files, options.RunnerLabel, RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(), Environment.ProcessorCount,
            GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, "runtime_default_not_configurable",
            "temperature_0_do_sample_false_package_search_defaults_otherwise", model.SystemPromptMode,
            prepared.Context.EffectiveMaxLength, prepared.Context.OriginalMaxLength,
            "resident_process_fresh_chat_session_per_request_full_locked_transcript_no_kv_reuse",
            options.MaxTokens, options.Warmups, options.Measurements, prepared.InitMilliseconds,
            prepared.CatalogMilliseconds, loadMilliseconds, metrics.MaximumObservedWorkingSetBytes,
            metrics.PeakPhysicalFootprintBytes, metrics.MemorySampleCount,
            [.. turns.Select(turn => new FoundryTurn(turn.Number, turn.System, turn.User, turn.LockedAssistant))],
            samples);
        await using var output = new FileStream(options.Output, FileMode.CreateNew, FileAccess.Write,
            FileShare.None);
        await JsonSerializer.SerializeAsync(output, evidence, FoundryJsonContext.Default.FoundryEvidence,
            cancellationToken).ConfigureAwait(false);
        Console.WriteLine(Path.GetFullPath(options.Output));
        return 0;
    }

    private static async Task<FoundryPreparedModel> PrepareAsync(FoundryRunOptions options,
        FoundryVariant pinned, int contextTokens, CancellationToken cancellationToken)
    {
        var phase = Stopwatch.StartNew();
        var manager = await FoundryRuntime.StartAsync(options.Cache, cancellationToken).ConfigureAwait(false);
        var initMilliseconds = phase.Elapsed.TotalMilliseconds;
        if (pinned.Device == "gpu")
        {
            await FoundryRuntime.RequireGpuProviderAsync(manager, options.Cache, allowDownload: false,
                cancellationToken).ConfigureAwait(false);
        }

        phase.Restart();
        var variant = await FoundryRuntime.ResolveAsync(manager, pinned, cancellationToken).ConfigureAwait(false);
        var catalogMilliseconds = phase.Elapsed.TotalMilliseconds;
        if (!await variant.IsCachedAsync(cancellationToken).ConfigureAwait(false))
        {
            throw new FoundryNotCachedException(
                $"Foundry variant '{pinned.Id}' is not cached in {Path.GetFullPath(options.Cache)}; run fetch first.");
        }

        var modelPath = await variant.GetPathAsync(cancellationToken).ConfigureAwait(false);
        var context = FoundryContextBound.Verify(modelPath, contextTokens);
        var files = await HashModelFilesAsync(modelPath, cancellationToken).ConfigureAwait(false);
        var genaiPath = Path.Combine(modelPath, "genai_config.json");
        var genai = File.Exists(genaiPath) ? JsonDocument.Parse(await File.ReadAllTextAsync(genaiPath,
            cancellationToken).ConfigureAwait(false)) : null;
        return new FoundryPreparedModel(variant, initMilliseconds, catalogMilliseconds, context, files, genai);
    }

    private static async Task<List<FoundrySample>> MeasureAsync(IModel variant, List<DialogueTurn> turns,
        FoundryRunOptions options, string systemPromptMode, Process process, CancellationToken cancellationToken)
    {
        var samples = new List<FoundrySample>();
        var rounds = options.Warmups + options.Measurements;
        for (var round = 0; round < rounds; round++)
        {
            for (var turn = 0; turn < turns.Count; turn++)
            {
                var plan = new FoundryRequestPlan(turn, round, round < options.Warmups, options.MaxTokens,
                    systemPromptMode);
                samples.Add(await FoundryRequest.MeasureAsync(variant, turns, plan, process, cancellationToken)
                    .ConfigureAwait(false));
                Console.Error.WriteLine($"foundry {options.Alias} turn {turn + 1}/{turns.Count} " +
                    $"round {round + 1}/{rounds} complete");
            }
        }

        return samples;
    }

    private static async Task<List<FoundryModelFile>> HashModelFilesAsync(string modelPath,
        CancellationToken cancellationToken)
    {
        var files = new List<FoundryModelFile>();
        foreach (var path in Directory.EnumerateFiles(modelPath, "*", SearchOption.AllDirectories)
            .Order(StringComparer.Ordinal))
        {
            files.Add(new FoundryModelFile(Path.GetRelativePath(modelPath, path).Replace('\\', '/'),
                new FileInfo(path).Length, await HashAsync(path, cancellationToken).ConfigureAwait(false)));
        }

        return files.Count > 0 ? files : throw new InvalidDataException($"No model files in {modelPath}.");
    }

    private static string? GenAiModelType(JsonDocument? genai) =>
        genai is not null && genai.RootElement.TryGetProperty("model", out var model) &&
        model.TryGetProperty("type", out var type) ? type.GetString() : null;

    private static JsonElement? GenAiSearch(JsonDocument? genai) =>
        genai is not null && genai.RootElement.TryGetProperty("search", out var search) ? search.Clone() : null;

    private static async Task<string> HashAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken)
            .ConfigureAwait(false));
    }
}

internal sealed record FoundryPreparedModel(IModel Variant, double InitMilliseconds, double CatalogMilliseconds,
    FoundryContextState Context, List<FoundryModelFile> Files, JsonDocument? GenAiConfig);

internal sealed record FoundryRunOptions(string Set, string Alias, string Cache, string Scenario,
    string Output, string Device, int MaxTokens, int Warmups, int Measurements, string RunnerLabel)
{
    public static FoundryRunOptions From(FoundryArguments arguments)
    {
        arguments.AllowOnly("--set", "--alias", "--cache", "--scenario", "--output", "--device",
            "--max-tokens", "--warmups", "--measurements", "--runner-label");
        return new FoundryRunOptions(arguments.Require("--set"), arguments.Require("--alias"),
            arguments.Require("--cache"), arguments.Require("--scenario"), arguments.Require("--output"),
            arguments.Device(), arguments.Count("--max-tokens", 64, 1, 512),
            arguments.Count("--warmups", 1, 0, 10), arguments.Count("--measurements", 3, 1, 30),
            arguments.Optional("--runner-label") ?? "local");
    }
}
