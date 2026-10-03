using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed class MemoryEvaluationState(string output)
{
    private bool _published;
    public string Status { get; set; } = "incomplete";
    public string? Error { get; set; }
    public MemoryEvaluationRequest? Request { get; set; }
    public OptimizationPackage? Package { get; set; }
    public List<MemoryPhase> Phases { get; } = [];
    public ObservedProcessMetrics? ProcessMetrics { get; set; }
    public double ProcessWallMilliseconds { get; set; }

    public async Task CaptureAsync(string name, Qwen2Model model, double milliseconds,
        MemoryGeneration? generation = null, OptimizationScore? score = null)
    {
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        Phases.Add(new MemoryPhase(name, model.AllocatedKvBytes, process.WorkingSet64,
            MacProcessFootprint.TryReadBytes(process.Id), GC.GetTotalMemory(forceFullCollection: false), milliseconds,
            model.RuntimeProfile, model.KernelImplementation, ModelGraphFingerprint.Compute(model.Graph).Value,
            generation, score));
        await WriteAsync().ConfigureAwait(false);
    }

    public async Task WriteAsync()
    {
        string[] limitations =
        [
            "KV bytes describe owned cache capacity; instantaneous process memory and sampled peaks include other allocations and file mappings.",
            "No forced garbage collection or operating-system cache flush occurs; disposal does not promise an immediate RSS reduction.",
            "Generation snapshots precede optional separately labeled scoring; scoring can reserve and contract KV too.",
            "Fresh comparators use identical IDs and the same backend/profile; comparisons diagnose retention math without qualifying a profile.",
            "Process wall includes verification, phase publication, repeated loads, generation and scoring; phase timers exclude provenance hashing.",
        ];
        var evidence = new MemoryEvaluationEvidence(1, "dynamic-kv-memory-diagnostic", Status, Error, DateTimeOffset.UtcNow,
            RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture.ToString(), RuntimeInformation.FrameworkDescription,
            Request, Package, [.. Phases], MemoryEvaluationComparisons.Build(Phases), ProcessWallMilliseconds, ProcessMetrics, limitations);
        var path = Path.GetFullPath(output);
        _ = Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var partial = path + ".partial-" + Guid.NewGuid().ToString("N");
        try
        {
            await File.WriteAllTextAsync(partial, JsonSerializer.Serialize(evidence, MemoryEvaluationJsonContext.Default.MemoryEvaluationEvidence),
                CancellationToken.None).ConfigureAwait(false);
            File.Move(partial, path, overwrite: _published);
            _published = true;
        }
        finally
        {
            File.Delete(partial);
        }
    }
}
