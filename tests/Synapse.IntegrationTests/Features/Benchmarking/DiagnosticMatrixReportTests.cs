using System.Diagnostics;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class DiagnosticMatrixReportTests
{
    [Test]
    public async Task RawMatrixEvidenceProducesDescriptivePerformanceReport()
    {
        var root = FindRepositoryRoot();
        var evidence = Path.Combine(root, "benchmarks", "results",
            "2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-four-subject-memory-clr-smoke.json");
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"));
        start.ArgumentList.Add("report");
        start.ArgumentList.Add("--input");
        start.ArgumentList.Add(evidence);
        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Matrix report process did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
        await Assert.That(process.ExitCode).IsEqualTo(0).Because(await error);
        var report = await output;
        await Assert.That(report).Contains("Performance diagnostic — no winner verdict");
        await Assert.That(report).Contains("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e");
        await Assert.That(report).Contains("dotllm");
        await Assert.That(report).Contains("llamacpp");
        await Assert.That(report).Contains("peak RSS MiB");
        await Assert.That(report).Contains("Measured rounds: 5");
    }
}
