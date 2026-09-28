using System.Diagnostics;
using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel]
public sealed class DiagnosticMatrixSmokeTests
{
    private const string ExpectedIds8 = "12095,13,1084,374,279,7772,3283,304";
    private const string ExpectedText8 = " Paris. It is the largest city in";
    private const string ExpectedIds32 = ExpectedIds8 +
        ",4505,323,279,2086,7772,304,279,1879,13,1084,374,7407,304,279,4126,315,279,8585,92900,11,304,279,9806,315";
    private const string ExpectedText32 =
        " Paris. It is the largest city in Europe and the second largest in the world. " +
        "It is located in the center of the French Alps, in the south of";

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
    public async Task DivergentLongContinuationIsIneligible()
    {
        using var evidence = await RunMatrixAsync(32, ExpectedIds32, ExpectedText32);
        var root = evidence.RootElement;
        await Assert.That(root.GetProperty("status").GetString()).IsEqualTo("ineligible_quality_mismatch");
        var samples = root.GetProperty("samples").EnumerateArray().ToArray();
        await Assert.That(samples.Length).IsEqualTo(4);
        var dotLlm = samples.Single(sample => sample.GetProperty("subject").GetString() == "dotllm");
        await Assert.That(dotLlm.GetProperty("quality_matched").GetBoolean()).IsFalse();
        await Assert.That(dotLlm.GetProperty("subject_result").GetProperty("text").GetString())
            .Contains("the south of France");
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
            "--synapse-executable", Path.Combine(root, "src", "Synapse.Cli", "bin", "Release", "net10.0", "synapse"),
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
