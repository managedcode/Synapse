using System.Diagnostics;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>The pass-key diagnostic runs the real CLI per prompt and records the answer check.</summary>
[NotInParallel]
public sealed class PasskeyCommandTests
{
    [Test]
    public async Task PasskeyCommandFindsKeyInShortPrompt()
    {
        var output = Path.Combine(Path.GetTempPath(), $"synapse-passkey-{Guid.NewGuid():N}.json");
        try
        {
            var (exitCode, error) = await RunAsync(
                "--prompt-tokens", "512", "--depths", "0.5", "--backend", "managed", "--context-size", "1024",
                "--output", output);

            await Assert.That(exitCode).IsEqualTo(0).Because(error);
            using var json = JsonDocument.Parse(await File.ReadAllTextAsync(output));
            var run = json.RootElement.GetProperty("runs")[0];
            await Assert.That(run.GetProperty("promptTokens").GetInt32()).IsBetween(480, 512);
            await Assert.That(run.GetProperty("key").GetString()).IsEqualTo("68317");
            await Assert.That(run.GetProperty("exactAnswer").GetBoolean()).IsTrue();
            await Assert.That(run.GetProperty("subject").GetString()).IsEqualTo("synapse-managed-simd-qwen2-q8_0xq8_0");
        }
        finally
        {
            File.Delete(output);
        }
    }

    [Test]
    public async Task PasskeyCommandRejectsPromptsLongerThanContext()
    {
        var output = Path.Combine(Path.GetTempPath(), $"synapse-passkey-{Guid.NewGuid():N}.json");

        var (exitCode, error) = await RunAsync(
            "--prompt-tokens", "2048", "--depths", "0.5", "--backend", "managed", "--context-size", "1024",
            "--output", output);

        await Assert.That(exitCode).IsEqualTo(2);
        await Assert.That(error).Contains("Usage: passkey");
        await Assert.That(File.Exists(output)).IsFalse();
    }

    private static async Task<(int ExitCode, string Error)> RunAsync(params string[] extra)
    {
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"),
            "passkey",
            "--synapse-executable", Path.Combine(AppContext.BaseDirectory, OperatingSystem.IsWindows() ? "synapse.exe" : "synapse"),
            "--model", ReferenceBenchmarkFixture.GetModelPath(),
            "--scenario", ReferenceBenchmarkFixture.BenchmarkInputPath("Scenarios", "passkey-qwen2.5.json"),
            "--threads", "4",
            .. extra,
        ];
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("The runner did not start.");
        var standardOutput = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(3));
        _ = await standardOutput;
        return (process.ExitCode, await error);
    }

}
