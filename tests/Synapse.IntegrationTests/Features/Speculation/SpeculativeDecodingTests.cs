using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.Speculation;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.Speculation;

/// <summary>Exact speculative decoding (ADR-020): a draft proposes, the target verifies, the output is the target's.</summary>
[NotInParallel]
public sealed class SpeculativeDecodingTests
{
    private static readonly TinyQwen2Gguf.Shape Target = new(Layers: 4, Heads: 7, KeyValueHeads: 1, HeadDimension: 64,
        FeedForward: 512, Vocabulary: 640, Context: 2_048);
    private static readonly int[] Prompt = [.. Enumerable.Range(0, 120).Select(index => index * 37 % 640)];

    [Test]
    [Arguments(2, 31)]
    [Arguments(4, 30)]
    public async Task SpeculativeOutputEqualsTargetGreedy(int draftLayers, int draftSeed)
    {
        var target = TinyQwen2Gguf.Write(Target, seed: 30);
        var draft = TinyQwen2Gguf.Write(Target with { Layers = draftLayers }, seed: draftSeed);
        try
        {
            using var targetModel = Load(target, KernelBackend.Managed);
            using var draftModel = Load(draft, KernelBackend.Managed);

            var expected = targetModel.Generate(Prompt, 40);
            var speculative = SpeculativeDecoding.Generate(targetModel, draftModel, Prompt, 40, draftTokens: 5);

            await Assert.That(speculative.Result.GeneratedTokens).IsEquivalentTo(expected.GeneratedTokens);
            await Assert.That(speculative.AcceptedTokens).IsLessThanOrEqualTo(speculative.DraftedTokens);
            if (draftSeed == 30)
            {
                // The draft is the target: every draft is accepted, so 39 tokens after the first take 7 passes.
                await Assert.That(speculative.AcceptedTokens).IsEqualTo(speculative.DraftedTokens);
                await Assert.That(speculative.TargetPasses).IsEqualTo(7);
            }
        }
        finally
        {
            File.Delete(target);
            File.Delete(draft);
        }
    }

    [Test]
    public async Task SpeculationRejectsIncompatibleModels()
    {
        var target = TinyQwen2Gguf.Write(Target, seed: 30);
        var wide = TinyQwen2Gguf.Write(Target with { Vocabulary = 704 }, seed: 31);
        try
        {
            using var targetModel = Load(target, KernelBackend.Managed);
            using var reference = Load(target, KernelBackend.Reference);
            using var wideDraft = Load(wide, KernelBackend.Managed);
            using var twin = Load(target, KernelBackend.Managed);

            await Assert.That(() => SpeculativeDecoding.Generate(targetModel, wideDraft, Prompt, 8, 4)).Throws<NotSupportedException>();
            await Assert.That(() => SpeculativeDecoding.Generate(reference, targetModel, Prompt, 8, 4)).Throws<NotSupportedException>();
            await Assert.That(() => SpeculativeDecoding.Generate(targetModel, twin, Prompt, 8, 0)).Throws<ArgumentOutOfRangeException>();
            await Assert.That(() => SpeculativeDecoding.Generate(targetModel, twin, Prompt, 8, 64)).Throws<ArgumentOutOfRangeException>();
            await Assert.That(() => SpeculativeDecoding.Generate(targetModel, targetModel, Prompt, 8, 4)).Throws<ArgumentException>();
        }
        finally
        {
            File.Delete(target);
            File.Delete(wide);
        }
    }

    [Test]
    public async Task SpeculativeMatchesTargetOnMetal()
    {
        GpuHardware.RequireMetal();
        var target = TinyQwen2Gguf.Write(Target with { HeadDimension = 128 }, seed: 40);
        var draft = TinyQwen2Gguf.Write(Target with { HeadDimension = 128, Layers = 2 }, seed: 41);
        try
        {
            using var targetModel = Load(target, KernelBackend.Metal);
            using var draftModel = Load(draft, KernelBackend.Metal);

            var expected = targetModel.Generate(Prompt, 32);
            var speculative = SpeculativeDecoding.Generate(targetModel, draftModel, Prompt, 32, draftTokens: 6);

            await Assert.That(speculative.Result.GeneratedTokens).IsEquivalentTo(expected.GeneratedTokens);
        }
        finally
        {
            File.Delete(target);
            File.Delete(draft);
        }
    }

    [Test]
    public async Task QwenHalfBillionSharesTheSevenBillionVocabulary()
    {
        var sevenBillion = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(ReferenceBenchmarkFixture.GetModelPath()))!,
            "qwen2.5-7b-instruct-1m-q8_0", "Qwen2.5-7B-Instruct-1M-Q8_0.gguf");
        Skip.Unless(File.Exists(sevenBillion), "not_run_missing_model: fetch the long-context set (`synapse model fetch --set long-context`).");
        using var draft = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        using var target = GgufFile.Open(sevenBillion);
        var draftTokens = draft.ReadStringArray("tokenizer.ggml.tokens").Length;

        await Assert.That(TokenIdentity.Digest(target, draftTokens)).IsEqualTo(TokenIdentity.Digest(draft, draftTokens));
    }

    private static Qwen2Model Load(string path, KernelBackend backend) => (Qwen2Model)ModelLoader.Load(
        path,
        new ModelLoadOptions { ContextSize = 1_024, MaximumParallelism = 8, KernelBackend = backend });
}
