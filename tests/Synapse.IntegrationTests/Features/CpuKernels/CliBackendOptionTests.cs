using System.Diagnostics;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

[NotInParallel]
public sealed class CliBackendOptionTests
{
    [Test]
    [Arguments("reference", "synapse-reference-qwen2-q8_0")]
    [Arguments("managed", "synapse-managed-simd-qwen2-q8_0xq8_0")]
    [Arguments("native", "synapse-native-rust-qwen2-q8_0xq8_0")]
    public async Task CliBackendOptionSelectsRuntimeProfile(string backend, string subject)
    {
        var (exitCode, output, error) = await RunCliAsync("--backend", backend);

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("subject").GetString()).IsEqualTo(subject);
        await Assert.That(json.RootElement.GetProperty("kernel_backend").GetString()).IsEqualTo(backend);
        await Assert.That(json.RootElement.GetProperty("generated_tokens").EnumerateArray()
            .Select(token => token.GetInt32()).ToArray()).IsEquivalentTo([12095, 13]);
    }

    [Test]
    public async Task CliRejectsUnknownBackend()
    {
        var (exitCode, _, error) = await RunCliAsync("--backend", "gpu-magic");

        await Assert.That(exitCode).IsEqualTo(2);
        await Assert.That(error).Contains("--backend <reference|managed|native|metal|cuda>");
    }

    [Test]
    [Arguments("managed")]
    [Arguments("native")]
    public async Task CliRunsConcurrentRequestsThroughOneModel(string backend)
    {
        var (exitCode, output, error) = await RunCliAsync("--backend", backend, "--concurrent-requests", "3");

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        var root = json.RootElement;
        var requests = root.GetProperty("requests").EnumerateArray().ToArray();
        await Assert.That(root.GetProperty("concurrent_requests").GetInt32()).IsEqualTo(3);
        await Assert.That(requests.Length).IsEqualTo(3);
        await Assert.That(requests.All(request => request.GetProperty("generated_tokens").EnumerateArray()
            .Select(token => token.GetInt32()).SequenceEqual([12095, 13]))).IsTrue();
        await Assert.That(root.GetProperty("total_generated_tokens").GetInt32()).IsEqualTo(6);
        await Assert.That(root.GetProperty("aggregate_output_tokens_per_second").GetDouble()).IsGreaterThan(0);
    }

    [Test]
    public async Task CliRejectsInvalidConcurrency()
    {
        var (exitCode, _, error) = await RunCliAsync("--concurrent-requests", "0");

        await Assert.That(exitCode).IsEqualTo(2);
        await Assert.That(error).Contains("[--concurrent-requests <count>]");
    }

    internal static async Task<(int ExitCode, string Output, string Error)> RunCliAsync(params string[] extra)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "synapse.dll"),
            "generate",
            "--model", await ReferenceBenchmarkFixture.GetCompiledModelPathAsync(),
            "--tokens", string.Join(',', ReferenceBenchmarkFixture.PromptTokens),
            "--max-tokens", "2",
            "--context-size", "64",
            "--threads", "2",
            .. extra,
        ];
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("The Synapse CLI process did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(2));
        return (process.ExitCode, await output, await error);
    }
}
