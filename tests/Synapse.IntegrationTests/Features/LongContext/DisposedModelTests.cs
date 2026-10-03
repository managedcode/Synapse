using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>REQ-CTX-010: closing the fenced owner rejects later work before allocation or mapped-weight access.</summary>
[NotInParallel]
public sealed class DisposedModelTests
{
    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task DisposedModelRejectsDirectWorkBeforeKvReservation(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        _ = model.Generate([1, 2, 3], 4);
        model.Dispose();

        await Assert.That(() => model.RunDirect(executor =>
        {
            executor.Reserve(64);
            return 0;
        })).Throws<ObjectDisposedException>();
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(0);
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task DisposedModelRejectsRequestsBeforePromptValidation(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        model.Dispose();

        await Assert.That(() => model.Generate([], 4)).Throws<ObjectDisposedException>();
        await Assert.That(() => model.EvaluatePromptLogits([])).Throws<ObjectDisposedException>();
        await Assert.That(async () => await model.GenerateAsync([], 4, CancellationToken.None)).Throws<ObjectDisposedException>();
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(0);
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task ConcurrentDisposalIsIdempotentAndRejectsValidRequests(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        _ = model.Generate([1, 2, 3], 4);

        await Task.WhenAll(Task.Run(model.Dispose), Task.Run(model.Dispose));

        await Assert.That(() => model.Generate([1, 2, 3], 4)).Throws<ObjectDisposedException>();
        await Assert.That(() => model.EvaluatePromptLogits([1, 2, 3])).Throws<ObjectDisposedException>();
        await Assert.That(async () => await model.GenerateAsync([1, 2, 3], 4, CancellationToken.None)).Throws<ObjectDisposedException>();
        await Assert.That(model.CreateTokenizer).Throws<ObjectDisposedException>();
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(0);
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task ReentrantDisposalIsRejectedAndOwnerRemainsUsable(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        using var cold = fixture.Load(backend);

        await Assert.That(() => model.Generate([1, 2, 3], 4, new DisposeFromProgress(model))).Throws<InvalidOperationException>();
        var actual = model.Generate([1, 2, 3], 4);
        var expected = cold.Generate([1, 2, 3], 4);

        await Assert.That(actual.GeneratedTokens.SequenceEqual(expected.GeneratedTokens)).IsTrue();
        await Assert.That(model.AllocatedKvBytes).IsGreaterThan(0);
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task AdmissionAndDisposalCompleteOrRejectWithoutOrphaningRequests(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        using var cold = fixture.Load(backend);
        var expected = cold.Generate([1, 2, 3], 4).GeneratedTokens;
        var admitted = AttemptAsync(model);
        var attempts = Enumerable.Range(0, 16).Select(_ => Task.Run(() => AttemptAsync(model))).Prepend(admitted).ToArray();
        var disposal = Task.Run(model.Dispose);

        var results = await Task.WhenAll(attempts).WaitAsync(TimeSpan.FromSeconds(10));
        await disposal.WaitAsync(TimeSpan.FromSeconds(10));

        foreach (var result in results.Where(result => result is not null))
        {
            await Assert.That(result!.SequenceEqual(expected)).IsTrue();
        }

        await Assert.That(model.AllocatedKvBytes).IsEqualTo(0);
        await Assert.That(async () => await model.GenerateAsync([1, 2, 3], 4, CancellationToken.None)).Throws<ObjectDisposedException>();
    }

    private static async Task<int[]?> AttemptAsync(Qwen2Model model)
    {
        try
        {
            return [.. (await model.GenerateAsync([1, 2, 3], 4, CancellationToken.None)).GeneratedTokens];
        }
        catch (ObjectDisposedException)
        {
            return null;
        }
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task SchedulerCallbackRejectsSelfJoinAndOwnerRemainsUsable(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        using var cold = fixture.Load(backend);
        var observed = new TaskCompletionSource<(bool Guarded, bool Rejected)>(TaskCreationOptions.RunContinuationsAsynchronously);
        model.BatchStepCompleted += trace =>
        {
            _ = trace;
            var guarded = model.IsExecutionCallback;
            var rejected = false;
            if (guarded)
            {
                try
                {
                    model.Dispose();
                }
                catch (InvalidOperationException)
                {
                    rejected = true;
                }
            }

            _ = observed.TrySetResult((guarded, rejected));
        };

        _ = await model.GenerateAsync([1, 2, 3], 4, CancellationToken.None);
        var (guarded, rejected) = await observed.Task.WaitAsync(TimeSpan.FromSeconds(10));
        var actual = model.Generate([1, 2, 3], 4);
        var expected = cold.Generate([1, 2, 3], 4);

        await Assert.That(guarded).IsTrue().Because("The real scheduler callback must advertise its self-join boundary before calling Dispose.");
        await Assert.That(rejected).IsTrue();
        await Assert.That(actual.GeneratedTokens.SequenceEqual(expected.GeneratedTokens)).IsTrue();
    }

    [Test]
    [Arguments(KernelBackend.Reference, false)]
    [Arguments(KernelBackend.Reference, true)]
    [Arguments(KernelBackend.Managed, false)]
    [Arguments(KernelBackend.Managed, true)]
    [Arguments(KernelBackend.Native, false)]
    [Arguments(KernelBackend.Native, true)]
    [Arguments(KernelBackend.Metal, false)]
    [Arguments(KernelBackend.Metal, true)]
    public async Task NestedDirectMathIsRejectedBeforeMutatingOuterKv(KernelBackend backend, bool score)
    {
        RequireHardware(backend);
        using var fixture = await DynamicKvContractionTests.PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        using var cold = fixture.Load(backend);
        int[] prompt = [.. Enumerable.Range(0, 257).Select(index => index % 64)];
        var caller = new NestedMathFromProgress(model, score);

        var actual = model.Generate(prompt, 4, caller);
        var expected = cold.Generate(prompt, 4);

        await Assert.That(caller.Rejected).IsTrue();
        await Assert.That(actual.GeneratedTokens.SequenceEqual(expected.GeneratedTokens)).IsTrue();
    }

    private static void RequireHardware(KernelBackend backend)
    {
        if (backend == KernelBackend.Metal)
        {
            _ = GpuHardware.RequireMetal();
        }
    }
}

/// <summary>A real caller that attempts disposal from progress and stops if an old implementation accepts it.</summary>
internal sealed class DisposeFromProgress(Qwen2Model model) : IProgress<GenerationProgress>
{
    public void Report(GenerationProgress value)
    {
        _ = value;
        model.Dispose();
        throw new OperationCanceledException("The caller stops before any access after unexpectedly accepted disposal.");
    }
}

/// <summary>A real progress consumer; an accepted nested call aborts before the corrupted outer request resumes.</summary>
internal sealed class NestedMathFromProgress(Qwen2Model model, bool score) : IProgress<GenerationProgress>
{
    public bool Rejected { get; private set; }

    public void Report(GenerationProgress value)
    {
        _ = value;
        try
        {
            if (score)
            {
                _ = model.Score([1, 2, 3], 1, progress: null);
            }
            else
            {
                _ = model.Generate([1, 2, 3], 4);
            }
        }
        catch (InvalidOperationException)
        {
            Rejected = true;
            return;
        }

        throw new OperationCanceledException("The caller stops before resuming outer math after an accepted nested request.");
    }
}
