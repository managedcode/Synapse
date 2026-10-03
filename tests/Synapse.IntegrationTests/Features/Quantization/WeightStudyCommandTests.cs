using System.Diagnostics;
using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

public sealed class WeightStudyCommandTests
{
    // TEST-QNT-006-9: a real tensor diagnostic cannot silently qualify model quality.
    [Test]
    public async Task RealWeightStudyRetainsRawSamplesAndUnqualifiedQualityStatus()
    {
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll"), "weight-study",
            "--model", Path.Combine(FindRepositoryRoot(), "artifacts", "models", "smollm2-135m-instruct-bf16", "model.safetensors"),
            "--tensor", "model.layers.0.self_attn.q_proj.weight", "--samples", "30",
        ];
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("Weight study did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(60));
        await Assert.That(process.ExitCode).IsEqualTo(0).Because(await error);
        using var evidence = JsonDocument.Parse(await output);
        var root = evidence.RootElement;
        await Assert.That(root.GetProperty("quality_status").GetString()).IsEqualTo("not_evaluated_synthetic_activations");
        await Assert.That(root.GetProperty("runtime_eligible").GetBoolean()).IsFalse();
        await Assert.That(root.GetProperty("samples").GetArrayLength()).IsEqualTo(30);
        await Assert.That(root.GetProperty("mean_distortion").GetProperty("relative").GetDouble()).IsGreaterThan(0.1);
        await Assert.That(root.GetProperty("model_sha256").GetString()!.Length).IsEqualTo(64);
    }
}
