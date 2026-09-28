using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

[NotInParallel]
public sealed class MetalBackendTests
{
    private static readonly int[] ExpectedContinuation = [12095, 13, 1084, 374, 279, 7772, 3283, 304];
    private static readonly int[][] Prompts =
    [
        [785, 6722, 315, 9625, 374],
        [785, 6722, 315, 9625, 374, 12095, 13],
        [1084, 374, 279],
        [9625, 374],
        [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279, 7772],
    ];

    [Test]
    public async Task MetalDeviceProbeReportsAppleGpu()
    {
        var device = GpuHardware.RequireMetal();

        await Assert.That(device.Backend).IsEqualTo("metal");
        await Assert.That(device.Name).Contains("Apple");
        await Assert.That(device.UnifiedMemory).IsTrue();
        await Assert.That(device.RecommendedWorkingSetBytes).IsGreaterThan(0);
        await Assert.That(device.MaximumBufferBytes).IsGreaterThan(1L << 30);
    }

    [Test]
    public async Task MetalBackendMatchesPinnedContinuation()
    {
        GpuHardware.RequireMetal();
        using var model = Load(contextSize: 64);

        var result = model.Generate(ReferenceBenchmarkFixture.PromptTokens, ExpectedContinuation.Length);

        await Assert.That(model.RuntimeProfile).IsEqualTo("metal-qwen2-q8_0xf32");
        await Assert.That(model.KernelImplementation).StartsWith("metal-apple");
        await Assert.That(result.GeneratedTokens).IsEquivalentTo(ExpectedContinuation);
    }

    [Test]
    public async Task MetalLogitsTrackReferenceLogits()
    {
        GpuHardware.RequireMetal();
        using var reference = (Qwen2Model)ModelLoader.Load(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions { ContextSize = 64, MaximumParallelism = 4, KernelBackend = KernelBackend.Reference });
        using var metal = Load(contextSize: 64);
        int[] prompt = [.. ReferenceBenchmarkFixture.PromptTokens, .. ExpectedContinuation[..3]];

        var expected = reference.EvaluatePromptLogits(prompt);
        var actual = metal.EvaluatePromptLogits(prompt);

        await Assert.That(ArgMax(actual)).IsEqualTo(ArgMax(expected));
        await Assert.That(LargestError(expected, actual)).IsLessThanOrEqualTo(Range(expected) * 0.002f);
    }

    [Test]
    public async Task MetalIncrementalDecodeTracksFullPrefill()
    {
        GpuHardware.RequireMetal();
        using var model = Load(contextSize: 64);
        var prompt = ReferenceBenchmarkFixture.PromptTokens.ToArray();

        var incremental = model.EvaluateIncrementalLogits(prompt, ExpectedContinuation[..6]);
        var full = model.EvaluatePromptLogits([.. prompt, .. ExpectedContinuation[..6]]);

        await Assert.That(ArgMax(incremental)).IsEqualTo(ArgMax(full));
        await Assert.That(LargestError(full, incremental)).IsLessThanOrEqualTo(Range(full) * 0.001f);
    }

    [Test]
    public async Task MetalConcurrentRequestsMatchIndependent()
    {
        GpuHardware.RequireMetal();
        using var model = Load(contextSize: 64, maximumSessions: 5);
        var independent = Prompts.Select(prompt => model.Generate(prompt, 10).GeneratedTokens).ToArray();

        var concurrent = await Task.WhenAll(Prompts.Select(prompt =>
            Task.Run(() => model.GenerateAsync(prompt, 10, CancellationToken.None))));

        for (var index = 0; index < Prompts.Length; index++)
        {
            await Assert.That(concurrent[index].GeneratedTokens).IsEquivalentTo(independent[index])
                .Because($"request {index}");
        }
    }

    [Test]
    public async Task MissingGpuLibraryFailsExplicitly()
    {
        var directory = Directory.CreateTempSubdirectory("synapse-no-gpu-");
        try
        {
            var exception = await Assert.That(() => NativeGpuLibrary.Load(directory.FullName))
                .Throws<NotSupportedException>();
            await Assert.That(exception!.Message).Contains(NativeGpuLibrary.LibraryFileName);
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    [Test]
    public async Task CudaBackendFailsExplicitlyWithoutDriver()
    {
        Skip.Unless(OperatingSystem.IsMacOS(), "The CUDA negative path is asserted on macOS, where no NVIDIA driver exists.");

        var probe = await Assert.That(() => GpuDevices.Probe(KernelBackend.Cuda)).Throws<NotSupportedException>();
        var load = await Assert.That(() => ModelLoader.Load(
                ReferenceBenchmarkFixture.GetModelPath(),
                new ModelLoadOptions { ContextSize = 64, KernelBackend = KernelBackend.Cuda }))
            .Throws<NotSupportedException>();

        await Assert.That(probe!.Message).Contains("CUDA");
        await Assert.That(load!.Message).Contains("cuda");
    }

    internal static Qwen2Model Load(int contextSize, int maximumSessions = 4) =>
        (Qwen2Model)ModelLoader.Load(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions
            {
                ContextSize = contextSize,
                MaximumParallelism = 4,
                KernelBackend = KernelBackend.Metal,
                MaximumConcurrentSessions = maximumSessions,
            });

    internal static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());

    internal static float LargestError(float[] expected, float[] actual) =>
        expected.Zip(actual, (left, right) => Math.Abs(left - right)).Max();

    internal static float Range(float[] values) => values.Max() - values.Min();
}
