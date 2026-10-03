using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.Quantization;

/// <summary>
/// GGUF K-quant weights (ADR-021), laid out like Q4_K_M: Q4_K for most matrices, Q6_K for V, the feed-forward down
/// projection, and the output. The oracle is the same model with every K-quant matrix stored as its exact FP32
/// decode.
/// </summary>
[NotInParallel]
public sealed class KQuantWeightTests
{
    private static readonly TinyQwen2Gguf.Shape Shape = new(Layers: 2, Heads: 2, KeyValueHeads: 1, HeadDimension: 128,
        FeedForward: 512, Vocabulary: 320, Context: 2_048);
    private static readonly int[] Prompt = [.. Enumerable.Range(0, 300).Select(index => index * 37 % 320)];
    private static readonly int[] Continuation = [5, 77, 300, 12];

    internal static TinyEncoding Q4KM(string tensor) =>
        tensor.EndsWith("attn_v.weight", StringComparison.Ordinal) || tensor.EndsWith("ffn_down.weight", StringComparison.Ordinal) ||
        tensor == "output.weight"
            ? TinyEncoding.Q6K
            : TinyEncoding.Q4K;

    [Test]
    public async Task KQuantReferenceEqualsDequantizedFp32()
    {
        var quantized = TinyQwen2Gguf.Write(Shape, seed: 51, encodings: Q4KM);
        var decoded = TinyQwen2Gguf.Write(Shape, seed: 51, encodings: Q4KM, dequantized: true);
        try
        {
            using var kQuant = Load(quantized, KernelBackend.Reference, KvCachePrecision.Fp32);
            using var oracle = Load(decoded, KernelBackend.Reference, KvCachePrecision.Fp32);

            var expected = oracle.EvaluateIncrementalLogits(Prompt, Continuation);
            var actual = kQuant.EvaluateIncrementalLogits(Prompt, Continuation);

            await Assert.That(actual).IsEquivalentTo(expected);
            await Assert.That(new FileInfo(quantized).Length).IsLessThan(new FileInfo(decoded).Length / 4);
        }
        finally
        {
            File.Delete(quantized);
            File.Delete(decoded);
        }
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    public async Task KQuantWeightsFailExplicitlyOnCpuKernels(KernelBackend backend)
    {
        var path = TinyQwen2Gguf.Write(Shape, seed: 52, encodings: Q4KM);
        try
        {
            var exception = await Assert.That(() => Load(path, backend, KvCachePrecision.Fp32)).Throws<NotSupportedException>();

            await Assert.That(exception!.Message).Contains("ADR-021");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    [Arguments(KvCachePrecision.Fp32, 0.002f)]
    [Arguments(KvCachePrecision.Fp16, 0.01f)]
    public async Task MetalMatchesReferenceOnKQuantWeights(KvCachePrecision precision, float tolerance)
    {
        GpuHardware.RequireMetal();
        var path = TinyQwen2Gguf.Write(Shape, seed: 53, encodings: Q4KM);
        try
        {
            using var reference = Load(path, KernelBackend.Reference, KvCachePrecision.Fp32);
            using var metal = Load(path, KernelBackend.Metal, precision);

            var expectedPrompt = reference.EvaluatePromptLogits(Prompt);
            var actualPrompt = metal.EvaluatePromptLogits(Prompt);
            var expected = reference.EvaluateIncrementalLogits(Prompt, Continuation);
            var actual = metal.EvaluateIncrementalLogits(Prompt, Continuation);

            await Assert.That(metal.RuntimeProfile).StartsWith("metal-qwen2-q4_k+q6_kxf32");
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

    private static Qwen2Model Load(string path, KernelBackend backend, KvCachePrecision precision) => (Qwen2Model)ModelLoader.LoadSourceForValidation(
        path,
        new ModelLoadOptions
        {
            ContextSize = 2_048,
            MaximumParallelism = 8,
            KernelBackend = backend,
            KvCachePrecision = precision,
        });
}
