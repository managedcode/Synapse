using System.Diagnostics;
using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel]
public sealed class DiagnosticMatrixSmokeTests
{
    private const string ExpectedIds8 = "12095,13,1084,374,279,7772,3283,304";
    private const string ExpectedText8 = " Paris. It is the largest city in";

    [Test]
    public async Task RealFourSubjectRoundPreservesRawMemoryAndQuality()
    {
        using var evidence = await RunMatrixAsync(8, ExpectedIds8, ExpectedText8);
        var root = evidence.RootElement;
        await Assert.That(root.GetProperty("status").GetString())
            .IsEqualTo("measured_diagnostic_no_statistical_verdict");
        foreach (var name in new[] { "model_sha256", "synapse_binary_sha256",
            "dot_llm_binary_sha256", "llama_cpp_binary_sha256" })
        {
            await Assert.That(root.GetProperty(name).GetString()!.Length).IsEqualTo(64);
        }

        var samples = root.GetProperty("samples").EnumerateArray().ToArray();
        await Assert.That(samples.Length).IsEqualTo(4);
        await Assert.That(samples.Select(sample => sample.GetProperty("subject").GetString()!))
            .IsEquivalentTo(["synapse", "dotllm", "llamasharp", "llamacpp"]);
        foreach (var sample in samples)
        {
            await Assert.That(sample.GetProperty("peak_resident_bytes").GetInt64()).IsGreaterThan(0);
            await Assert.That(sample.GetProperty("memory_sample_count").GetInt32()).IsGreaterThan(0);
            await Assert.That(sample.GetProperty("quality_matched").GetBoolean()).IsTrue();
            if (OperatingSystem.IsMacOS())
            {
                await Assert.That(sample.GetProperty("peak_physical_footprint_bytes").GetInt64())
                    .IsGreaterThan(0);
            }
        }

        var synapse = samples.Single(sample => sample.GetProperty("subject").GetString() == "synapse");
        await Assert.That(synapse.GetProperty("subject_result")
            .GetProperty("managed_live_heap_after_generation_bytes").GetInt64()).IsGreaterThan(0);
    }

    [Test]
    public async Task IncorrectReferenceIsIneligible()
    {
        // A deliberately impossible expected token exercises the quality gate
        // without turning the verification suite into a long performance run.
        using var evidence = await RunMatrixAsync(8, ExpectedIds8[..^3] + "2147483647", ExpectedText8);
        var root = evidence.RootElement;
        await Assert.That(root.GetProperty("status").GetString()).IsEqualTo("ineligible_quality_mismatch");
        var samples = root.GetProperty("samples").EnumerateArray().ToArray();
        await Assert.That(samples.Length).IsEqualTo(4);
        var synapse = samples.Single(sample => sample.GetProperty("subject").GetString() == "synapse");
        await Assert.That(synapse.GetProperty("quality_matched").GetBoolean()).IsFalse();
    }

    private static async Task<JsonDocument> RunMatrixAsync(int maxTokens, string expectedIds, string expectedText)
    {
        var output = Path.Combine(Path.GetTempPath(), $"synapse-matrix-{Guid.NewGuid():N}.json");
        try
        {
            var start = CreateStart(maxTokens, expectedIds, expectedText, output);
            using var process = Process.Start(start) ?? throw new InvalidOperationException("Matrix runner did not start.");
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(5));
            await Assert.That(process.ExitCode).IsEqualTo(0).Because(await stderr);
            await Assert.That(await stdout).Contains(output);
            return JsonDocument.Parse(await File.ReadAllTextAsync(output));
        }
        finally
        {
            if (File.Exists(output))
            {
                File.Delete(output);
            }
        }
    }

    private static ProcessStartInfo CreateStart(int maxTokens, string expectedIds, string expectedText, string output)
    {
        var root = FindRepositoryRoot();
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"), "matrix",
            "--model", GetModelPath(), "--prompt", Prompt,
            "--prompt-token-ids", string.Join(',', PromptTokens),
            "--expected-token-ids", expectedIds, "--expected-text", expectedText,
            "--synapse-executable", Path.Combine(root, "src", "Synapse.Cli", "bin", "Release", "net10.0",
                OperatingSystem.IsWindows() ? "synapse.exe" : "synapse"),
            "--dotllm-executable", RequireEnvironmentFile("SYNAPSE_DOTLLM_EXECUTABLE"),
            "--dotllm-version", "d88040451d7db56e5dfef9d5754ad0955b0f7fe5",
            "--llamacpp-executable", RequireEnvironmentFile("SYNAPSE_LLAMACPP_EXECUTABLE"),
            "--llamacpp-version", "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4",
            "--max-tokens", maxTokens.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--threads", "8", "--warmups", "0", "--measurements", "1", "--output", output,
        ];
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        return start;
    }
}
