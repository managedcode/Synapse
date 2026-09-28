using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Cli.Features.TextGeneration;

internal static class GenerationCommand
{
    public static async Task<int> RunAsync(IReadOnlyList<string> arguments)
    {
        var options = GenerationOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine(
                "Usage: synapse generate --model <model.gguf> --tokens <id,id,...> " +
                "[--max-tokens <count>] [--context-size <count>] [--threads <count>] " +
                $"[--backend <{CpuKernelBackendNames.Usage}>] [--concurrent-requests <count>]");
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
            using var model = ModelLoader.Load(
                options.ModelPath,
                new ModelLoadOptions
                {
                    ContextSize = options.ContextSize,
                    MaximumParallelism = options.Threads,
                    KernelBackend = options.Backend,
                });
            loadTimer.Stop();
            process.Refresh();
            var workingSetAfterLoad = process.WorkingSet64;
            var managedHeapAfterLoad = GC.GetTotalMemory(forceFullCollection: false);
            var result = model.Generate(options.Tokens, options.MaximumTokens);
            subjectTimer.Stop();
            var managedHeapAfterGeneration = GC.GetTotalMemory(forceFullCollection: false);
            var managedAllocated = GC.GetTotalAllocatedBytes() - managedAllocatedBefore;
            process.Refresh();
            var maximumObservedWorkingSet = Math.Max(workingSetAfterLoad, process.WorkingSet64);
            if (process.PeakWorkingSet64 > 0)
            {
                maximumObservedWorkingSet = Math.Max(maximumObservedWorkingSet, process.PeakWorkingSet64);
            }

            var subjectCpu = process.TotalProcessorTime - subjectCpuStart;
            var decodeSeconds = Math.Max(
                0.000_001,
                result.Elapsed.TotalSeconds - result.TimeToFirstToken.TotalSeconds);
            var output = new GenerationOutput(
                $"synapse-{model.RuntimeProfile}",
                CpuKernelBackendNames.ToName(options.Backend),
                model.KernelImplementation,
                Path.GetFullPath(options.ModelPath),
                result.PromptTokens,
                result.GeneratedTokens,
                options.Threads,
                loadTimer.Elapsed.TotalMilliseconds,
                result.TimeToFirstToken.TotalMilliseconds,
                result.Elapsed.TotalMilliseconds,
                result.Elapsed.TotalSeconds == 0
                    ? null
                    : result.GeneratedTokens.Count / result.Elapsed.TotalSeconds,
                result.GeneratedTokens.Count <= 1
                    ? null
                    : (result.GeneratedTokens.Count - 1) / decodeSeconds,
                subjectTimer.Elapsed.TotalMilliseconds,
                subjectCpu.TotalMilliseconds,
                subjectTimer.Elapsed.TotalSeconds == 0
                    ? null
                    : subjectCpu.TotalSeconds / subjectTimer.Elapsed.TotalSeconds,
                workingSetAfterLoad,
                maximumObservedWorkingSet,
                managedHeapAfterLoad,
                managedHeapAfterGeneration,
                managedAllocated);
            Console.WriteLine(JsonSerializer.Serialize(output, GenerationJsonContext.Default.GenerationOutput));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    internal sealed record GenerationOptions(
        string ModelPath,
        int[] Tokens,
        int MaximumTokens,
        int ContextSize,
        int Threads,
        CpuKernelBackend Backend,
        int ConcurrentRequests)
    {
        public static GenerationOptions? Parse(IReadOnlyList<string> arguments)
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var index = 0; index < arguments.Count; index += 2)
            {
                if (index + 1 >= arguments.Count ||
                    !arguments[index].StartsWith("--", StringComparison.Ordinal))
                {
                    return null;
                }

                values[arguments[index]] = arguments[index + 1];
            }

            if (!values.TryGetValue("--model", out var modelPath) ||
                !values.TryGetValue("--tokens", out var tokenText))
            {
                return null;
            }

            var tokens = ParseTokens(tokenText);
            var maximumTokens = ParsePositive(values.GetValueOrDefault("--max-tokens"), 1);
            var contextSize = ParsePositive(values.GetValueOrDefault("--context-size"), 512);
            var threads = ParsePositive(values.GetValueOrDefault("--threads"), Environment.ProcessorCount);
            var backend = CpuKernelBackend.Managed;
            if (values.TryGetValue("--backend", out var backendName) &&
                !CpuKernelBackendNames.TryParse(backendName, out backend))
            {
                return null;
            }

            var concurrentRequests = ParsePositive(values.GetValueOrDefault("--concurrent-requests"), 1);
            return tokens is { Length: > 0 } && maximumTokens > 0 && contextSize > 0 && threads > 0 &&
                concurrentRequests is > 0 and <= 64
                ? new GenerationOptions(modelPath, tokens, maximumTokens, contextSize, threads, backend, concurrentRequests)
                : null;
        }

        private static int[]? ParseTokens(string value)
        {
            var parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var tokens = new int[parts.Length];
            for (var index = 0; index < parts.Length; index++)
            {
                if (!int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out tokens[index]) ||
                    tokens[index] < 0)
                {
                    return null;
                }
            }

            return tokens;
        }

        private static int ParsePositive(string? value, int fallback) => value is null
                ? fallback
                : int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                    ? parsed
                    : -1;
    }
}

internal sealed record GenerationOutput(
    string Subject,
    string KernelBackend,
    string KernelImplementation,
    string ModelPath,
    IReadOnlyList<int> PromptTokens,
    IReadOnlyList<int> GeneratedTokens,
    int Threads,
    double LoadMilliseconds,
    double TimeToFirstTokenMilliseconds,
    double TotalGenerationMilliseconds,
    double? TotalOutputTokensPerSecond,
    double? DecodeTokensPerSecond,
    double SubjectWallMilliseconds,
    double ProcessCpuMilliseconds,
    double? AverageCpuCores,
    long WorkingSetAfterLoadBytes,
    long MaximumObservedWorkingSetBytes,
    long ManagedLiveHeapAfterLoadBytes,
    long ManagedLiveHeapAfterGenerationBytes,
    long ManagedAllocatedDuringSubjectBytes);

[JsonSerializable(typeof(GenerationOutput))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class GenerationJsonContext : JsonSerializerContext;
