using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Cli.Features.LayerDrop;
using ManagedCode.Synapse.Cli.Features.Speculation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Cli.Features.TextGeneration;

internal static class GenerationCommand
{
    public static async Task<int> RunAsync(IReadOnlyList<string> arguments)
    {
        var options = GenerationOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine(GenerationOptions.Usage);
            return 2;
        }

        return options.ConcurrentRequests > 1
            ? await ConcurrentGenerationCommand.RunAsync(options).ConfigureAwait(false)
            : RunSingle(options);
    }

    private static int RunSingle(GenerationOptions options)
    {
        try
        {
            using var process = Process.GetCurrentProcess();
            var subjectTimer = Stopwatch.StartNew();
            var subjectCpuStart = process.TotalProcessorTime;
            var managedAllocatedBefore = GC.GetTotalAllocatedBytes();
            var loadTimer = Stopwatch.StartNew();
            if (options.LayerDrop is { } drop)
            {
                Console.Error.WriteLine(LayerDropArguments.Describe(drop));
            }

            using var model = ModelLoader.Load(
                options.ModelPath,
                new ModelLoadOptions
                {
                    ContextSize = options.ContextSize,
                    MaximumParallelism = options.Threads,
                    KernelBackend = options.Backend,
                    MaximumConcurrentSessions = 1,
                    ScoringRowsPerStep = (options.Speculation?.DraftTokens ?? 0) + 1,
                    RopeScaling = options.RopeScaling,
                    KvCachePrecision = options.KvCachePrecision,
                    KvPageActivation = options.KvPages,
                    LayerDrop = options.LayerDrop,
                });
            loadTimer.Stop();
            process.Refresh();
            var workingSetAfterLoad = process.WorkingSet64;
            var managedHeapAfterLoad = GC.GetTotalMemory(forceFullCollection: false);
            var (result, speculation) = options.Speculation is { } speculative
                ? speculative.Run(model, options)
                : (model.Generate(options.Tokens, options.MaximumTokens, new StandardErrorProgress()), null);
            subjectTimer.Stop();
            process.Refresh();
            var measurement = new SubjectMeasurement(
                loadTimer.Elapsed,
                subjectTimer.Elapsed,
                process.TotalProcessorTime - subjectCpuStart,
                workingSetAfterLoad,
                Math.Max(Math.Max(workingSetAfterLoad, process.WorkingSet64), process.PeakWorkingSet64),
                managedHeapAfterLoad,
                GC.GetTotalMemory(forceFullCollection: false),
                GC.GetTotalAllocatedBytes() - managedAllocatedBefore);
            var output = CreateOutput(options, model, result, measurement) with { Speculation = speculation };
            Console.WriteLine(JsonSerializer.Serialize(output, GenerationJsonContext.Default.GenerationOutput));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static GenerationOutput CreateOutput(
        GenerationOptions options,
        ITextGenerationModel model,
        TextGenerationResult result,
        SubjectMeasurement measurement)
    {
        var decodeSeconds = Math.Max(0.000_001, result.Elapsed.TotalSeconds - result.TimeToFirstToken.TotalSeconds);
        return new GenerationOutput(
            $"synapse-{model.RuntimeProfile}",
            KernelBackendNames.ToName(options.Backend),
            model.KernelImplementation,
            Path.GetFullPath(options.ModelPath),
            result.PromptTokens,
            result.PromptTokens.Count,
            options.ContextSize,
            options.RopeScaling?.Name,
            result.GeneratedTokens,
            DecodeOrNull(model, result.GeneratedTokens),
            options.Threads,
            measurement.Load.TotalMilliseconds,
            result.TimeToFirstToken.TotalMilliseconds,
            result.Elapsed.TotalMilliseconds,
            result.TimeToFirstToken.TotalSeconds == 0 ? null : result.PromptTokens.Count / result.TimeToFirstToken.TotalSeconds,
            result.Elapsed.TotalSeconds == 0 ? null : result.GeneratedTokens.Count / result.Elapsed.TotalSeconds,
            result.GeneratedTokens.Count <= 1 ? null : (result.GeneratedTokens.Count - 1) / decodeSeconds,
            measurement.Wall.TotalMilliseconds,
            measurement.Cpu.TotalMilliseconds,
            measurement.Wall.TotalSeconds == 0 ? null : measurement.Cpu.TotalSeconds / measurement.Wall.TotalSeconds,
            measurement.WorkingSetAfterLoad,
            measurement.MaximumObservedWorkingSet,
            measurement.ManagedHeapAfterLoad,
            measurement.ManagedHeapAfterGeneration,
            measurement.ManagedAllocated)
        {
            RequestedOptimization = options.Optimization,
            RuntimeProfile = model.RuntimeProfile,
            AllocatedKvBytes = model is Qwen2Model qwen ? qwen.AllocatedKvBytes : null,
        };
    }

    /// <summary>Decodes with the loaded model's tokenizer (ADR-014); null when that tokenizer is not implemented.</summary>
    internal static string? DecodeOrNull(ITextGenerationModel model, IReadOnlyList<int> tokens)
    {
        try
        {
            return model.CreateTokenizer().Decode(tokens);
        }
        catch (Exception exception) when (exception is NotSupportedException or InvalidDataException)
        {
            return null;
        }
    }

    private sealed record SubjectMeasurement(
        TimeSpan Load,
        TimeSpan Wall,
        TimeSpan Cpu,
        long WorkingSetAfterLoad,
        long MaximumObservedWorkingSet,
        long ManagedHeapAfterLoad,
        long ManagedHeapAfterGeneration,
        long ManagedAllocated);
}

internal sealed record GenerationOutput(
    string Subject,
    string KernelBackend,
    string KernelImplementation,
    string ModelPath,
    IReadOnlyList<int> PromptTokens,
    int PromptTokenCount,
    int ContextSize,
    string? RopeScaling,
    IReadOnlyList<int> GeneratedTokens,
    string? GeneratedText,
    int Threads,
    double LoadMilliseconds,
    double TimeToFirstTokenMilliseconds,
    double TotalGenerationMilliseconds,
    double? PromptTokensPerSecondThroughFirstToken,
    double? TotalOutputTokensPerSecond,
    double? DecodeTokensPerSecond,
    double SubjectWallMilliseconds,
    double ProcessCpuMilliseconds,
    double? AverageCpuCores,
    long WorkingSetAfterLoadBytes,
    long MaximumObservedWorkingSetBytes,
    long ManagedLiveHeapAfterLoadBytes,
    long ManagedLiveHeapAfterGenerationBytes,
    long ManagedAllocatedDuringSubjectBytes)
{
    /// <summary>Draft and acceptance counts of a speculative run (ADR-020), or null.</summary>
    public SpeculationOutput? Speculation { get; init; }

    /// <summary>The explicitly requested control mode, independent of the effective numerical profile.</summary>
    public string RequestedOptimization { get; init; } = "custom";

    /// <summary>Effective loaded numerical profile.</summary>
    public string RuntimeProfile { get; init; } = string.Empty;

    /// <summary>KV bytes actually retained by the loaded executor after generation, when available.</summary>
    public long? AllocatedKvBytes { get; init; }
}

[JsonSerializable(typeof(GenerationOutput))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class GenerationJsonContext : JsonSerializerContext;

/// <summary>Writes prompt and output progress to standard error at most once per second, plus completion lines.</summary>
internal sealed class StandardErrorProgress : IProgress<GenerationProgress>
{
    private TimeSpan? _lastReport;
    private bool _prefillDone;

    public void Report(GenerationProgress value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var prefillCompleted = !_prefillDone && value.EvaluatedPromptTokens == value.PromptTokens;
        if (!prefillCompleted && _lastReport is { } last && value.Elapsed - last < TimeSpan.FromSeconds(1))
        {
            return;
        }

        _lastReport = value.Elapsed;
        _prefillDone |= prefillCompleted;
        Console.Error.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"progress: prefill {value.EvaluatedPromptTokens}/{value.PromptTokens}, generated {value.GeneratedTokens}, {value.Elapsed.TotalSeconds:F1} s"));
    }
}
