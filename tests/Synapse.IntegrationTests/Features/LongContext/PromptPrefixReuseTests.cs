using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>Prompt prefix reuse on the direct session (ADR-018): skip the shared prefix, keep the answer.</summary>
[NotInParallel]
public sealed class PromptPrefixReuseTests
{
    private static readonly int[] Document = [.. Enumerable.Range(0, 700).Select(index => (index * 7919 % 150_000) + 100)];
    private static readonly int[] Question = [785, 6722, 315, 9625, 374];

    [Test]
    public async Task ReusedPrefixSkipsSharedTokensAndMatchesAColdRunOnCpu()
    {
        using var warm = Load(KernelBackend.Managed, reuse: true);
        using var cold = Load(KernelBackend.Managed, reuse: false);

        var first = warm.Generate(Document, 4);
        var second = warm.Generate([.. Document, .. Question], 4);
        var expected = cold.Generate([.. Document, .. Question], 4);

        await Assert.That(first.ReusedPromptTokens).IsEqualTo(0);
        await Assert.That(second.ReusedPromptTokens).IsEqualTo(Document.Length);
        await Assert.That(second.GeneratedTokens).IsEquivalentTo(expected.GeneratedTokens);
        await Assert.That(cold.Generate([.. Document, .. Question], 4).ReusedPromptTokens).IsEqualTo(0);
    }

    [Test]
    public async Task ReusedPrefixMatchesAColdRunOnMetal()
    {
        GpuHardware.RequireMetal();
        using var warm = Load(KernelBackend.Metal, reuse: true);
        using var cold = Load(KernelBackend.Metal, reuse: false);

        _ = warm.Generate(Document, 4);
        var second = warm.Generate([.. Document, .. Question], 8);
        var expected = cold.Generate([.. Document, .. Question], 8);

        await Assert.That(second.ReusedPromptTokens).IsEqualTo(Document.Length);
        await Assert.That(second.GeneratedTokens).IsEquivalentTo(expected.GeneratedTokens);
    }

    [Test]
    public async Task IdenticalPromptReevaluatesOnlyItsLastTokenAndOtherDirectWorkInvalidates()
    {
        using var model = Load(KernelBackend.Managed, reuse: true);

        _ = model.Generate(Document, 2);
        var repeated = model.Generate(Document, 2);
        _ = model.EvaluatePromptLogits(Question);
        var afterOtherWork = model.Generate(Document, 2);

        await Assert.That(repeated.ReusedPromptTokens).IsEqualTo(Document.Length - 1);
        await Assert.That(afterOtherWork.ReusedPromptTokens).IsEqualTo(0);
    }

    [Test]
    public async Task FailedRequestKeepsOnlyThePrefixItDidNotOverwrite()
    {
        using var warm = Load(KernelBackend.Managed, reuse: true);
        using var cold = Load(KernelBackend.Managed, reuse: false);
        int[] diverging = [.. Document.Take(300), .. Question, .. Question];

        _ = warm.Generate(Document, 2);
        await Assert.That(() => warm.Generate(diverging, 16, new StopAfterTokens(3))).Throws<OperationCanceledException>();
        var retried = warm.Generate(Document, 4);
        var expected = cold.Generate(Document, 4);

        // The failed request wrote its own K and V from position 300 on; only the shared 300 stay valid.
        await Assert.That(retried.ReusedPromptTokens).IsEqualTo(300);
        await Assert.That(retried.GeneratedTokens).IsEquivalentTo(expected.GeneratedTokens);
    }

    private static Qwen2Model Load(KernelBackend backend, bool reuse) => (Qwen2Model)ModelLoader.LoadSourceForValidation(
        ReferenceBenchmarkFixture.GetModelPath(),
        new ModelLoadOptions
        {
            ContextSize = 1_024,
            MaximumParallelism = 8,
            KernelBackend = backend,
            ReusePromptPrefix = reuse,
        });
}

/// <summary>A caller that cancels a synchronous generation from its progress callback.</summary>
internal sealed class StopAfterTokens(int tokens) : IProgress<GenerationProgress>
{
    public void Report(GenerationProgress value)
    {
        if (value.GeneratedTokens >= tokens)
        {
            throw new OperationCanceledException("The caller stopped the generation.");
        }
    }
}
