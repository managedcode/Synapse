using System.Diagnostics;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

public sealed class DiagnosticMatrixBackendArgumentTests
{
    [Test]
    public async Task MatrixRejectsUnknownSynapseBackendBeforeMeasuring()
    {
        var output = Path.Combine(Path.GetTempPath(), $"synapse-matrix-{Guid.NewGuid():N}.json");
        var (exitCode, error) = await RunMatrixAsync("--synapse-backend", "gpu-magic", "--output", output);

        await Assert.That(exitCode).IsEqualTo(2);
        await Assert.That(error).Contains("[--synapse-backend reference|managed|native]");
        await Assert.That(File.Exists(output)).IsFalse();
    }

    private static async Task<(int ExitCode, string Error)> RunMatrixAsync(params string[] extra)
    {
        var existing = GetModelPath();
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"),
            "matrix",
            "--model", existing,
            "--prompt", Prompt,
            "--prompt-token-ids", string.Join(',', PromptTokens),
            "--expected-token-ids", "12095",
            "--expected-text", " Paris",
            "--synapse-executable", existing,
            "--dotllm-executable", existing,
            "--dotllm-version", "test",
            "--llamacpp-executable", existing,
            "--llamacpp-version", "test",
            "--max-tokens", "1",
            .. extra,
        ];
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("Matrix process did not start.");
        _ = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
        return (process.ExitCode, await error);
    }
}
