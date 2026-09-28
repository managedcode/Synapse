using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

[NotInParallel]
public sealed class CliMetalBackendTests
{
    [Test]
    public async Task CliMetalBackendReportsGpuProfile()
    {
        GpuHardware.RequireMetal();

        var (exitCode, output, error) = await CliBackendOptionTests.RunCliAsync("--backend", "metal");

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        var root = json.RootElement;
        await Assert.That(root.GetProperty("subject").GetString()).IsEqualTo("synapse-metal-qwen2-q8_0xf32");
        await Assert.That(root.GetProperty("kernel_backend").GetString()).IsEqualTo("metal");
        await Assert.That(root.GetProperty("kernel_implementation").GetString()).StartsWith("metal-apple");
        await Assert.That(root.GetProperty("generated_tokens").EnumerateArray()
            .Select(token => token.GetInt32()).ToArray()).IsEquivalentTo([12095, 13]);
    }

    [Test]
    public async Task CliMetalFp16KvReportsProfile()
    {
        GpuHardware.RequireMetal();

        var (exitCode, output, error) = await CliBackendOptionTests.RunCliAsync("--backend", "metal", "--kv-precision", "f16");
        var (cpuExit, _, cpuError) = await CliBackendOptionTests.RunCliAsync("--backend", "managed", "--kv-precision", "f16");

        await Assert.That(exitCode).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("subject").GetString()).IsEqualTo("synapse-metal-qwen2-q8_0xf32-kvf16");
        await Assert.That(cpuExit).IsEqualTo(1);
        await Assert.That(cpuError).Contains("FP16 KV");
    }
}
