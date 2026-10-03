using System.Runtime.InteropServices;
using System.Text.Json;

internal sealed class OptimizationEvaluationState(string output)
{
    private bool _published;
    public string Status { get; set; } = "incomplete";
    public string? Error { get; set; }
    public string PlanSha256 { get; set; } = string.Empty;
    public string? HaystackSha256 { get; set; }
    public OptimizationPlan? Plan { get; set; }
    public List<OptimizationPackage> Packages { get; } = [];
    public List<OptimizationSample> Samples { get; } = [];

    public async Task WriteAsync()
    {
        string[] limitations =
        [
            "Tail perplexity scores the bounded suffix of synthetic retrieval prompts; it is not held-out model quality qualification.",
            "Cold means a fresh process/model session, not flushed operating-system file caches; repeat is a separate controlled request.",
            "Process wall includes child provenance hashing/load/generation/scoring; load and generation timings are separately bounded.",
            "Observed process memory includes load, generation and scoring; it is not generation-only memory.",
            "Changed output lengths remain visible; these diagnostics do not automatically select or qualify a winner.",
        ];
        var evidence = new OptimizationEvidence(1, "paired-long-context-optimization-diagnostic", Status, DateTimeOffset.UtcNow,
            RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture.ToString(), RuntimeInformation.FrameworkDescription,
            PlanSha256, HaystackSha256, Plan, [.. Packages], [.. Samples],
            Plan is null ? [] : OptimizationComparisons.Build(Plan.BaselineProfile, Samples), limitations, Error);
        var path = Path.GetFullPath(output);
        _ = Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var partial = path + ".partial-" + Guid.NewGuid().ToString("N");
        try
        {
            await File.WriteAllTextAsync(partial, JsonSerializer.Serialize(evidence, OptimizationJsonContext.Default.OptimizationEvidence),
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
