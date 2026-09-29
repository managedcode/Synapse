using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

/// <summary>
/// The Metal kernels at every supported head dimension (64 in Qwen2.5-0.5B, 128 in Qwen2.5-7B) against the scalar
/// FP32 reference backend, on a tiny generated model with the 7B's grouping (seven query heads per KV head).
/// </summary>
[NotInParallel]
public sealed class MetalHeadDimensionTests
{
    [Test]
    [Arguments(64, KvCachePrecision.Fp32, 0.002f)]
    [Arguments(128, KvCachePrecision.Fp32, 0.002f)]
    [Arguments(128, KvCachePrecision.Fp16, 0.01f)]
    public async Task MetalMatchesReferenceAtEveryHeadDimension(int headDimension, KvCachePrecision precision, float tolerance)
    {
        GpuHardware.RequireMetal();
        var path = TinyQwen2Gguf.Write(
            new TinyQwen2Gguf.Shape(Layers: 2, Heads: 7, KeyValueHeads: 1, headDimension, FeedForward: 512, Vocabulary: 640, Context: 4_096),
            seed: headDimension);
        try
        {
            // 1,200 prompt tokens cross many 32-key tiles and prompt runs; 6 decode tokens split long key ranges.
            int[] prompt = [.. Enumerable.Range(0, 1_200).Select(index => index * 37 % 640)];
            int[] continuation = [5, 77, 300, 12, 639, 1];
            using var reference = Load(path, KernelBackend.Reference, KvCachePrecision.Fp32);
            using var metal = Load(path, KernelBackend.Metal, precision);

            var expectedPrompt = reference.EvaluatePromptLogits(prompt);
            var actualPrompt = metal.EvaluatePromptLogits(prompt);
            var expected = reference.EvaluateIncrementalLogits(prompt, continuation);
            var actual = metal.EvaluateIncrementalLogits(prompt, continuation);

            await Assert.That(MetalBackendTests.ArgMax(actualPrompt)).IsEqualTo(MetalBackendTests.ArgMax(expectedPrompt));
            await Assert.That(MetalBackendTests.LargestError(expectedPrompt, actualPrompt))
                .IsLessThanOrEqualTo(MetalBackendTests.Range(expectedPrompt) * tolerance);
            await Assert.That(MetalBackendTests.ArgMax(actual)).IsEqualTo(MetalBackendTests.ArgMax(expected));
            await Assert.That(MetalBackendTests.LargestError(expected, actual))
                .IsLessThanOrEqualTo(MetalBackendTests.Range(expected) * tolerance);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Qwen2Model Load(string path, KernelBackend backend, KvCachePrecision precision) => (Qwen2Model)ModelLoader.Load(
        path,
        new ModelLoadOptions
        {
            ContextSize = 4_096,
            MaximumParallelism = 8,
            KernelBackend = backend,
            KvCachePrecision = precision,
        });
}
