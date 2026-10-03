using System.Diagnostics;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class DialogueReportTests
{
    [Test]
    public async Task CpuDialogueEvidenceReportsGrowingContextWithoutCacheClaim()
    {
        var report = await RenderAsync(
            "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-3turn-diagnostic.json");
        await Assert.That(report).Contains("locked_transcript_fresh_process_per_turn_no_kv_reuse");
        await Assert.That(report).Contains("|3|synapse|");
        await Assert.That(report).Contains("|3|dotllm|");
        await Assert.That(report).Contains("|3|llamasharp|");
        await Assert.That(report).Contains("|3|llamacpp|");
        await Assert.That(report).Contains("No KV or prefix-cache reuse is measured");
        await Assert.That(report).Contains("Process CPU ms");
        await Assert.That(report).Contains("Avg CPU cores");
        await Assert.That(report).Contains("Peak RSS MiB");
    }

    [Test]
    public async Task MlxDialogueEvidenceReportsObservedCacheHitsAndSeparateMemoryScope()
    {
        var report = await RenderAsync(
            "2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-3turn-diagnostic.json");
        await Assert.That(report).Contains("swiftlm-mlx-metal");
        await Assert.That(report).Contains("Cache-hit tokens come from the pinned server's log");
        await Assert.That(report).Contains("First fresh-server request");
        await Assert.That(report).Contains("Avg CPU cores");
        await Assert.That(report).Contains("77.0");
        await Assert.That(report).Contains("178.0");
        await Assert.That(report).Contains("Do not rank this cohort against CPU GGUF");
    }

    private static async Task<string> RenderAsync(string file)
    {
        var evidence = RecordedBenchmarkPath(file);
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"));
        start.ArgumentList.Add("report-dialogue");
        start.ArgumentList.Add("--input");
        start.ArgumentList.Add(evidence);
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Dialogue report did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
        await Assert.That(process.ExitCode).IsEqualTo(0).Because(await error);
        return await output;
    }
}
