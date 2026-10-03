using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class OptimizationWorkerCommand
{
    public static Task<int> RunAsync(string[] args) => OptimizationRunSupport.WithConsoleCancellationAsync(token => RunAsync(args, token));

    internal static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken)
    {
        if (!OptimizationRunSupport.Arguments(args, "--request", out var input, out var output))
        {
            await Console.Error.WriteLineAsync("Usage: optimization-worker --request <json> --output <new.json>").ConfigureAwait(false);
            return 2;
        }

        OptimizationRequest? request = null;
        var state = new WorkerState();
        if (File.Exists(output))
        {
            await Console.Error.WriteLineAsync("Worker output already exists.").ConfigureAwait(false);
            return 1;
        }

        try
        {
            if (new FileInfo(input).Length > 32 * 1024 * 1024)
            {
                throw new InvalidDataException("Worker request exceeds 32 MiB.");
            }

            request = JsonSerializer.Deserialize(await File.ReadAllTextAsync(input, cancellationToken).ConfigureAwait(false),
                OptimizationJsonContext.Default.OptimizationRequest) ?? throw new InvalidDataException("Empty worker request.");
            ValidateRequest(request);
            var actualPackage = await OptimizationValidation.DescribePackageAsync(request.ModelPath, cancellationToken).ConfigureAwait(false);
            if (actualPackage != request.Package)
            {
                throw new InvalidDataException("Worker package provenance does not match the actual prepared package.");
            }

            state.Package = actualPackage;
            Execute(request, state, cancellationToken);
            state.Status = "completed";
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            state.Status = FailureStatus(exception, request);
            state.Error = exception.ToString();
            await Console.Error.WriteLineAsync(exception.Message).ConfigureAwait(false);
        }

        var result = state.Result(request);
        _ = Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        await using (var publication = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, FileOptions.Asynchronous))
        {
            await JsonSerializer.SerializeAsync(publication, result, OptimizationJsonContext.Default.OptimizationWorkerResult,
                CancellationToken.None).ConfigureAwait(false);
        }
        return state.Status == "completed" ? 0 : state.Status == "cancelled" ? 130 : 1;
    }

    private static void ValidateRequest(OptimizationRequest request)
    {
        _ = OptimizationValidation.PreparedPath(request.ModelPath, Environment.CurrentDirectory);
        OptimizationValidation.ValidateProfile(request.Profile);
        OptimizationValidation.ValidateRope(request.RopeScaling);
        if (request.SchemaVersion != 1 || request.Package is null || request.PromptTokens is not { Length: >= 2 and <= 1_048_576 } ||
            request.PromptTokens.Any(token => token < 0) || request.MaximumNewTokens is < 1 or > 4096 ||
            (long)request.PromptTokens.Length + request.MaximumNewTokens > request.ContextSize ||
            request.ContextSize is < 2 or > 1_048_576 || request.ScoredTailTokens is < 1 or > 4096 || request.Threads is < 1 or > 256 ||
            request.Answers is not { Length: > 0 and <= 128 } || request.Answers.Any(answer => string.IsNullOrEmpty(answer) || answer.Length > 1024) ||
            Path.GetFullPath(request.Package.Path) != Path.GetFullPath(request.ModelPath) || request.Package.FileSha256.Length != 64 ||
            request.Package.Identity.Length != 64 || request.Package.SourceSha256.Length != 64 || request.Package.GraphFingerprint.Length != 64)
        {
            throw new InvalidDataException("Worker request needs validated package provenance, bounded tokens, answers and context/output limits.");
        }
    }

    private static string FailureStatus(Exception exception, OptimizationRequest? request) => exception switch
    {
        OperationCanceledException => "cancelled",
        NotSupportedException when (request?.Profile?.Backend is "metal" or "cuda") &&
            exception.Message.Contains("device", StringComparison.OrdinalIgnoreCase) &&
            (exception.Message.Contains("unavailable", StringComparison.OrdinalIgnoreCase) ||
                exception.Message.Contains("not found", StringComparison.OrdinalIgnoreCase)) => "not_run_missing_hardware",
        _ => "failed",
    };

    private static void Execute(OptimizationRequest request, WorkerState state, CancellationToken cancellationToken)
    {
        var load = Stopwatch.StartNew();
        using var model = (Qwen2Model)ModelLoader.Load(request.ModelPath, LoadOptions(request));
        load.Stop();
        state.LoadMilliseconds = load.Elapsed.TotalMilliseconds;
        state.RuntimeProfile = model.RuntimeProfile;
        state.KernelImplementation = model.KernelImplementation;
        state.GraphFingerprint = ModelGraphFingerprint.Compute(model.Graph).Value;
        var tokenizer = model.CreateTokenizer();
        string[] turns = request.IncludeRepeatedPrompt ? ["cold", "repeat"] : ["cold"];
        foreach (var turn in turns)
        {
            var progress = new WorkerProgress(cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            var generation = model.Generate(request.PromptTokens, request.MaximumNewTokens, progress);
            var text = tokenizer.Decode(generation.GeneratedTokens);
            var found = request.Answers.Count(answer => text.Contains(answer, StringComparison.Ordinal));
            var decode = generation.Elapsed - generation.TimeToFirstToken;
            state.Generations.Add(new OptimizationGeneration(turn, generation.PromptTokens.Count, request.MaximumNewTokens, [.. generation.GeneratedTokens], text,
                generation.ReusedPromptTokens, progress.PrefillMilliseconds, generation.TimeToFirstToken.TotalMilliseconds, generation.Elapsed.TotalMilliseconds,
                decode > TimeSpan.Zero && generation.GeneratedTokens.Count > 1 ? (generation.GeneratedTokens.Count - 1) / decode.TotalSeconds : null,
                (double)found / request.Answers.Length, found == request.Answers.Length, model.AllocatedKvBytes));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var first = Math.Max(0, request.PromptTokens.Length - 1 - request.ScoredTailTokens);
        var scores = model.Score(request.PromptTokens, first, new WorkerProgress(cancellationToken));
        state.Score = OptimizationScoreFactory.Create(scores);
    }

    private static ModelLoadOptions LoadOptions(OptimizationRequest request) => new()
    {
        ContextSize = request.ContextSize,
        MaximumParallelism = request.Threads,
        MaximumConcurrentSessions = 1,
        KernelBackend = KernelBackendNames.TryParse(request.Profile.Backend, out var backend) ? backend : throw new InvalidDataException("Unknown backend."),
        KvCachePrecision = request.Profile.KvPrecision == "f16" ? KvCachePrecision.Fp16 : KvCachePrecision.Fp32,
        ReusePromptPrefix = request.Profile.ReusePromptPrefix,
        KvPageActivation = request.Profile.KvPages?.ToRuntime(),
        LayerDrop = request.Profile.DropLayers is { } layers ? new LayerDropProfile(layers) { EvidenceSha256 = request.Profile.EvidenceSha256 } : null,
        RopeScaling = request.RopeScaling,
        ScoringRowsPerStep = 1,
    };

    private sealed class WorkerState
    {
        public OptimizationPackage? Package { get; set; }
        public string Status { get; set; } = "failed";
        public string? Error { get; set; }
        public double? LoadMilliseconds { get; set; }
        public string? RuntimeProfile { get; set; }
        public string? KernelImplementation { get; set; }
        public string? GraphFingerprint { get; set; }
        public List<OptimizationGeneration> Generations { get; } = [];
        public OptimizationScore? Score { get; set; }

        public OptimizationWorkerResult Result(OptimizationRequest? request) => new(Status, Error, Environment.ProcessId,
            Package?.Identity, Package?.FileSha256, Package?.SourceSha256,
            OptimizationRunSupport.TokenSha256(request?.PromptTokens ?? []), GraphFingerprint, RuntimeProfile, KernelImplementation,
            LoadMilliseconds, [.. Generations], Score);
    }

    private sealed class WorkerProgress(CancellationToken cancellationToken) : IProgress<GenerationProgress>
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
                    $"optimization-worker pid={Environment.ProcessId} prompt={value.EvaluatedPromptTokens}/{value.PromptTokens} generated={value.GeneratedTokens} elapsed={value.Elapsed.TotalSeconds:F1}s"));
            }
        }
    }
}
