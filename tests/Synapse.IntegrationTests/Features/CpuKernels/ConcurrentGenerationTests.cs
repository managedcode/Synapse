using System.Collections.Concurrent;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

[NotInParallel]
public sealed class ConcurrentGenerationTests
{
    private static readonly int[][] Prompts =
    [
        [785, 6722, 315, 9625, 374],
        [785, 6722, 315, 9625, 374, 12095, 13],
        [1084, 374, 279],
        [9625, 374],
        [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279, 7772],
    ];

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Reference)]
    public async Task RaggedBatchMatchesIndependent(KernelBackend backend)
    {
        var newTokens = backend == KernelBackend.Reference ? 3 : 10;
        using var model = Load(backend, maximumSessions: 5);
        var independent = Prompts.Select(prompt => model.Generate(prompt, newTokens).GeneratedTokens).ToArray();

        var concurrent = await Task.WhenAll(Prompts.Select(prompt =>
            Task.Run(() => model.GenerateAsync(prompt, newTokens, CancellationToken.None))));

        for (var index = 0; index < Prompts.Length; index++)
        {
            await Assert.That(concurrent[index].GeneratedTokens).IsEquivalentTo(independent[index])
                .Because($"request {index}");
            await Assert.That(concurrent[index].PromptTokens).IsEquivalentTo(Prompts[index]);
            await Assert.That(concurrent[index].TimeToFirstToken).IsGreaterThan(TimeSpan.Zero);
        }
    }

    [Test]
    public async Task LongPrefillDoesNotStarveDecode()
    {
        var shortPrompt = Prompts[0];
        int[] longPrompt = [.. Enumerable.Range(0, 150).Select(index => Prompts[1][index % Prompts[1].Length])];
        using var model = Load(KernelBackend.Managed, maximumSessions: 2, contextSize: 192);
        var steps = new ConcurrentQueue<BatchStepTrace>();
        model.BatchStepCompleted += steps.Enqueue;

        var shortRun = model.GenerateAsync(shortPrompt, 12, CancellationToken.None);
        var longRun = model.GenerateAsync(longPrompt, 2, CancellationToken.None);
        _ = await Task.WhenAll(shortRun, longRun);

        var contested = steps.Where(step =>
            step.Entries.Any(entry => entry.PromptLength == longPrompt.Length && entry.PrefillTokens > 0) &&
            step.Entries.Any(entry => entry.PromptLength == shortPrompt.Length && entry.GeneratedBefore is > 0 and < 12))
            .ToArray();
        await Assert.That(contested.Length).IsGreaterThanOrEqualTo(2);
        await Assert.That(contested.All(step => step.Entries
            .Single(entry => entry.PromptLength == shortPrompt.Length).DecodeTokens == 1)).IsTrue();
        await Assert.That(steps.All(step => step.Entries.Sum(entry => entry.PrefillTokens + entry.DecodeTokens) <= 64))
            .IsTrue();
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    public async Task CancelledSlotReusedSafely(KernelBackend backend)
    {
        using var model = Load(backend, maximumSessions: 1);
        var expected = model.Generate(Prompts[2], 6).GeneratedTokens;
        using var cancellation = new CancellationTokenSource();
        var slots = new ConcurrentDictionary<int, int>();
        model.BatchStepCompleted += step =>
        {
            foreach (var entry in step.Entries)
            {
                slots[entry.PromptLength] = entry.Slot;
                if (entry.PromptLength == Prompts[4].Length && entry.GeneratedBefore >= 3)
                {
                    cancellation.Cancel();
                }
            }
        };

        var cancelled = model.GenerateAsync(Prompts[4], 50, cancellation.Token);
        var queued = model.GenerateAsync(Prompts[2], 6, CancellationToken.None);

        await Assert.That(async () => await cancelled).Throws<OperationCanceledException>();
        var reused = await queued;
        await Assert.That(reused.GeneratedTokens).IsEquivalentTo(expected);
        await Assert.That(slots[Prompts[2].Length]).IsEqualTo(slots[Prompts[4].Length]);
    }

    private static Qwen2Model Load(KernelBackend backend, int maximumSessions, int contextSize = 64) =>
        (Qwen2Model)ModelLoader.Load(
            ReferenceBenchmarkFixture.GetModelPath(),
            new ModelLoadOptions
            {
                ContextSize = contextSize,
                MaximumParallelism = 3,
                KernelBackend = backend,
                MaximumConcurrentSessions = maximumSessions,
            });
}
