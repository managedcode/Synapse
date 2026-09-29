using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Cli.Features.LayerDrop;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Cli.Features.TextGeneration;

/// <summary>
/// Runs N identical requests at once through one loaded model (ADR-007) and reports per-request latency plus
/// aggregate output throughput. Diagnostic only; paired benchmark rules still apply to any claim.
/// </summary>
internal static class ConcurrentGenerationCommand
{
    public static async Task<int> RunAsync(GenerationOptions options)
    {
        try
        {
            using var process = Process.GetCurrentProcess();
            var subjectTimer = Stopwatch.StartNew();
            var cpuStart = process.TotalProcessorTime;
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
                    MaximumConcurrentSessions = options.ConcurrentRequests,
                    RopeScaling = options.RopeScaling,
                    KvCachePrecision = options.KvCachePrecision,
                    KvPageActivation = options.KvPages,
                    LayerDrop = options.LayerDrop,
                });
            loadTimer.Stop();
            var wallTimer = Stopwatch.StartNew();
            var results = await Task.WhenAll(Enumerable.Range(0, options.ConcurrentRequests).Select(_ =>
                model.GenerateAsync(options.Tokens, options.MaximumTokens, CancellationToken.None))).ConfigureAwait(false);
            wallTimer.Stop();
            subjectTimer.Stop();
            process.Refresh();
            Console.WriteLine(JsonSerializer.Serialize(
                CreateOutput(options, model, results, loadTimer.Elapsed, wallTimer.Elapsed, subjectTimer.Elapsed,
                    process.TotalProcessorTime - cpuStart, Math.Max(process.WorkingSet64, process.PeakWorkingSet64)),
                ConcurrentGenerationJsonContext.Default.ConcurrentGenerationOutput));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static ConcurrentGenerationOutput CreateOutput(
        GenerationOptions options,
        ITextGenerationModel model,
        TextGenerationResult[] results,
        TimeSpan load,
        TimeSpan wall,
        TimeSpan subject,
        TimeSpan cpu,
        long workingSet)
    {
        var generated = results.Sum(result => result.GeneratedTokens.Count);
        return new ConcurrentGenerationOutput(
            $"synapse-{model.RuntimeProfile}",
            KernelBackendNames.ToName(options.Backend),
            model.KernelImplementation,
            Path.GetFullPath(options.ModelPath),
            options.Tokens,
            options.Threads,
            options.ConcurrentRequests,
            load.TotalMilliseconds,
            wall.TotalMilliseconds,
            generated,
            wall.TotalSeconds == 0 ? null : generated / wall.TotalSeconds,
            subject.TotalMilliseconds,
            cpu.TotalMilliseconds,
            subject.TotalSeconds == 0 ? null : cpu.TotalSeconds / subject.TotalSeconds,
            workingSet,
            [.. results.Select(result => new ConcurrentRequestOutput(
                result.GeneratedTokens,
                result.TimeToFirstToken.TotalMilliseconds,
                result.Elapsed.TotalMilliseconds))]);
    }
}

internal sealed record ConcurrentRequestOutput(
    IReadOnlyList<int> GeneratedTokens,
    double TimeToFirstTokenMilliseconds,
    double TotalGenerationMilliseconds);

internal sealed record ConcurrentGenerationOutput(
    string Subject,
    string KernelBackend,
    string KernelImplementation,
    string ModelPath,
    IReadOnlyList<int> PromptTokens,
    int Threads,
    int ConcurrentRequests,
    double LoadMilliseconds,
    double ConcurrentWallMilliseconds,
    int TotalGeneratedTokens,
    double? AggregateOutputTokensPerSecond,
    double SubjectWallMilliseconds,
    double ProcessCpuMilliseconds,
    double? AverageCpuCores,
    long MaximumObservedWorkingSetBytes,
    IReadOnlyList<ConcurrentRequestOutput> Requests);

[JsonSerializable(typeof(ConcurrentGenerationOutput))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class ConcurrentGenerationJsonContext : JsonSerializerContext;
