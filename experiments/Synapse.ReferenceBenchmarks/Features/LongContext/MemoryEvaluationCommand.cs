using System.Diagnostics;
using System.Text.Json;

internal static class MemoryEvaluationCommand
{
    public static Task<int> RunAsync(string[] args) => OptimizationRunSupport.WithConsoleCancellationAsync(token => RunAsync(args, token));

    internal static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken)
    {
        if (!OptimizationRunSupport.Arguments(args, "--request", out var input, out var output))
        {
            await Console.Error.WriteLineAsync("Usage: memory-eval --request <json> --output <new.json>").ConfigureAwait(false);
            return 2;
        }

        if (File.Exists(output))
        {
            await Console.Error.WriteLineAsync("Memory evaluation output already exists.").ConfigureAwait(false);
            return 1;
        }

        var state = new MemoryEvaluationState(output);
        var wall = Stopwatch.StartNew();
        using var process = Process.GetCurrentProcess();
        await using var sampler = new ProcessMemorySampler(process);
        try
        {
            state.Request = await ReadRequestAsync(input, cancellationToken).ConfigureAwait(false);
            state.Package = await OptimizationValidation.DescribePackageAsync(state.Request.ModelPath, cancellationToken).ConfigureAwait(false);
            await MemoryEvaluationExecution.RunAsync(state, cancellationToken).ConfigureAwait(false);
            state.Status = "completed";
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            state.Status = FailureStatus(exception, state.Request);
            state.Error = exception.ToString();
            await Console.Error.WriteLineAsync(exception.Message).ConfigureAwait(false);
        }

        state.ProcessMetrics = await sampler.CompleteAsync().ConfigureAwait(false);
        state.ProcessWallMilliseconds = wall.Elapsed.TotalMilliseconds;
        await state.WriteAsync().ConfigureAwait(false);
        return state.Status == "completed" ? 0 : state.Status == "cancelled" ? 130 : 1;
    }

    private static async Task<MemoryEvaluationRequest> ReadRequestAsync(string path, CancellationToken cancellationToken)
    {
        if (new FileInfo(path).Length > 32 * 1024 * 1024)
        {
            throw new InvalidDataException("Memory request exceeds 32 MiB.");
        }

        var request = JsonSerializer.Deserialize(await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false),
            MemoryEvaluationJsonContext.Default.MemoryEvaluationRequest) ?? throw new InvalidDataException("Empty memory request.");
        var directory = Path.GetDirectoryName(Path.GetFullPath(path))!;
        request = request with { ModelPath = OptimizationValidation.PreparedPath(request.ModelPath, directory) };
        OptimizationValidation.ValidateProfile(request.Profile);
        OptimizationValidation.ValidateRope(request.RopeScaling);
        if (request.Profile.ModelPath is not null)
        {
            throw new InvalidDataException("Memory profile modelPath is not accepted; use the request modelPath.");
        }

        if (request.SchemaVersion != 1 || request.ContextSize is < 2 or > 1_048_576 || request.MaximumNewTokens is < 1 or > 4096 ||
            request.ScoredTailTokens is < 0 or > 4096 || request.Threads is < 1 or > 256 ||
            request.ShortPromptTokens is not { Length: >= 2 and <= 1_048_576 } ||
            request.LongPromptTokens is not { Length: >= 2 and <= 1_048_576 } ||
            request.LongPromptTokens.Length <= request.ShortPromptTokens.Length ||
            request.ShortPromptTokens.Any(token => token < 0) || request.LongPromptTokens.Any(token => token < 0) ||
            (long)request.LongPromptTokens.Length + request.MaximumNewTokens > request.ContextSize)
        {
            throw new InvalidDataException("Memory request needs bounded short/long tokens, context, output and scoring limits.");
        }

        return request;
    }

    private static string FailureStatus(Exception exception, MemoryEvaluationRequest? request) => exception switch
    {
        OperationCanceledException => "cancelled",
        NotSupportedException when (request?.Profile?.Backend is "metal" or "cuda") &&
            exception.Message.Contains("device", StringComparison.OrdinalIgnoreCase) &&
            (exception.Message.Contains("unavailable", StringComparison.OrdinalIgnoreCase) ||
                exception.Message.Contains("not found", StringComparison.OrdinalIgnoreCase)) => "not_run_missing_hardware",
        _ => "failed",
    };
}
