using System.Diagnostics;
using System.Globalization;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class MemoryEvaluationExecution
{
    public static async Task RunAsync(MemoryEvaluationState state, CancellationToken cancellationToken)
    {
        var request = state.Request!;
        await RunGrowingModelAsync(state, cancellationToken).ConfigureAwait(false);
        await RunFreshModelAsync(state, "fresh-short", request.ShortPromptTokens, cancellationToken).ConfigureAwait(false);
        await RunFreshModelAsync(state, "fresh-long", request.LongPromptTokens, cancellationToken).ConfigureAwait(false);
    }

    private static async Task RunGrowingModelAsync(MemoryEvaluationState state, CancellationToken cancellationToken)
    {
        var request = state.Request!;
        var model = Load(request, cancellationToken, out var milliseconds);
        try
        {
            await state.CaptureAsync("loaded", model, milliseconds).ConfigureAwait(false);
            await GenerateAsync(state, model, "short-first", request.ShortPromptTokens, cancellationToken).ConfigureAwait(false);
            await GenerateAsync(state, model, "long", request.LongPromptTokens, cancellationToken).ConfigureAwait(false);
            await GenerateAsync(state, model, "short-after-long", request.ShortPromptTokens, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await DisposeAsync(state, "disposed", model).ConfigureAwait(false);
        }
    }

    private static async Task RunFreshModelAsync(MemoryEvaluationState state, string name, int[] tokens, CancellationToken cancellationToken)
    {
        var model = Load(state.Request!, cancellationToken, out var milliseconds);
        try
        {
            await state.CaptureAsync(name + "-loaded", model, milliseconds).ConfigureAwait(false);
            await GenerateAsync(state, model, name, tokens, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await DisposeAsync(state, name + "-disposed", model).ConfigureAwait(false);
        }
    }

    private static async Task GenerateAsync(MemoryEvaluationState state, Qwen2Model model, string name, int[] tokens,
        CancellationToken cancellationToken)
    {
        var request = state.Request!;
        cancellationToken.ThrowIfCancellationRequested();
        var progress = new MemoryProgress(name, cancellationToken);
        var generated = model.Generate(tokens, request.MaximumNewTokens, progress);
        var generation = new MemoryGeneration(OptimizationRunSupport.TokenSha256(tokens), tokens.Length, request.MaximumNewTokens,
            [.. generated.GeneratedTokens], generated.ReusedPromptTokens, progress.PrefillMilliseconds,
            generated.TimeToFirstToken.TotalMilliseconds, generated.Elapsed.TotalMilliseconds);
        await state.CaptureAsync(name, model, generated.Elapsed.TotalMilliseconds, generation).ConfigureAwait(false);
        if (request.ScoredTailTokens > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var first = Math.Max(0, tokens.Length - 1 - request.ScoredTailTokens);
            var scores = model.Score(tokens, first, new MemoryProgress(name + "-score", cancellationToken));
            var score = OptimizationScoreFactory.Create(scores);
            await state.CaptureAsync(name + "-score", model, scores.Elapsed.TotalMilliseconds, score: score).ConfigureAwait(false);
        }
    }

    private static Qwen2Model Load(MemoryEvaluationRequest request, CancellationToken cancellationToken, out double milliseconds)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var timer = Stopwatch.StartNew();
        var model = (Qwen2Model)ModelLoader.Load(request.ModelPath, new ModelLoadOptions
        {
            ContextSize = request.ContextSize,
            MaximumParallelism = request.Threads,
            MaximumConcurrentSessions = 1,
            ScoringRowsPerStep = 1,
            KernelBackend = KernelBackendNames.TryParse(request.Profile.Backend, out var backend) ? backend : throw new InvalidDataException("Unknown backend."),
            KvCachePrecision = request.Profile.KvPrecision == "f16" ? KvCachePrecision.Fp16 : KvCachePrecision.Fp32,
            ReusePromptPrefix = request.Profile.ReusePromptPrefix,
            KvPageActivation = request.Profile.KvPages?.ToRuntime(),
            LayerDrop = request.Profile.DropLayers is { } layers ? new LayerDropProfile(layers) { EvidenceSha256 = request.Profile.EvidenceSha256 } : null,
            RopeScaling = request.RopeScaling,
        });
        milliseconds = timer.Elapsed.TotalMilliseconds;
        return model;
    }

    private static async Task DisposeAsync(MemoryEvaluationState state, string name, Qwen2Model model)
    {
        var timer = Stopwatch.StartNew();
        model.Dispose();
        await state.CaptureAsync(name, model, timer.Elapsed.TotalMilliseconds).ConfigureAwait(false);
    }

    private sealed class MemoryProgress(string name, CancellationToken cancellationToken) : IProgress<GenerationProgress>
    {
        private TimeSpan _last;
        public double? PrefillMilliseconds { get; private set; }

        public void Report(GenerationProgress value)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (value.GeneratedTokens == 0 && value.EvaluatedPromptTokens == value.PromptTokens)
            {
                PrefillMilliseconds = value.Elapsed.TotalMilliseconds;
            }
            if (value.Elapsed - _last >= TimeSpan.FromSeconds(1))
            {
                _last = value.Elapsed;
                Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"memory-eval phase={name} prompt={value.EvaluatedPromptTokens}/{value.PromptTokens} generated={value.GeneratedTokens} elapsed={value.Elapsed.TotalSeconds:F1}s"));
            }
        }
    }
}
