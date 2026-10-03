using ManagedCode.Synapse.IntegrationTests.Features.Quantization;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>Partial K-quant GEMM tiles and chunk tails execute real mixed-encoding model math against FP32.</summary>
[NotInParallel]
public sealed class MetalKQuantGemmTests
{
    [Test]
    [Arguments(17, KvCachePrecision.Fp32, false, 0.002f)]
    [Arguments(513, KvCachePrecision.Fp32, false, 0.002f)]
    [Arguments(513, KvCachePrecision.Fp16, false, 0.01f)]
    [Arguments(17, KvCachePrecision.Fp32, true, 0.002f)]
    public async Task KQuantGemmPartialTilesAndChunkTailTrackReference(
        int promptTokens, KvCachePrecision precision, bool allQ4, float tolerance)
    {
        GpuHardware.RequireMetal();
        var shape = new TinyQwen2Gguf.Shape(Layers: 2, Heads: 4, KeyValueHeads: 2, HeadDimension: 128,
            FeedForward: 768, Vocabulary: 323, Context: 1_024);
        var path = TinyQwen2Gguf.Write(shape, seed: 58,
            encodings: allQ4 ? _ => TinyEncoding.Q4K : KQuantWeightTests.Q4KM);
        try
        {
            int[] prompt = [.. Enumerable.Range(0, promptTokens).Select(index => index * 37 % shape.Vocabulary)];
            int[] continuation = [0, 322, 31];
            using var reference = Load(path, KernelBackend.Reference, KvCachePrecision.Fp32);
            using var metal = Load(path, KernelBackend.Metal, precision);

            await AssertLogits(reference.EvaluatePromptLogits(prompt), metal.EvaluatePromptLogits(prompt), tolerance);
            await AssertLogits(reference.EvaluateIncrementalLogits(prompt, continuation),
                metal.EvaluateIncrementalLogits(prompt, continuation), tolerance);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static async Task AssertLogits(float[] expected, float[] actual, float tolerance)
    {
        await Assert.That(MetalBackendTests.ArgMax(actual)).IsEqualTo(MetalBackendTests.ArgMax(expected));
        await Assert.That(MetalBackendTests.LargestError(expected, actual))
            .IsLessThanOrEqualTo(MetalBackendTests.Range(expected) * tolerance);
    }

    private static Qwen2Model Load(string path, KernelBackend backend, KvCachePrecision precision) =>
        (Qwen2Model)ModelLoader.LoadSourceForValidation(
            path,
            new ModelLoadOptions
            {
                ContextSize = 1_024,
                MaximumParallelism = 2,
                KernelBackend = backend,
                KvCachePrecision = precision,
            });
}
