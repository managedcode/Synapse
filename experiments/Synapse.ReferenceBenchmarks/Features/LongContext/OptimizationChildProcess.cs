using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;

internal sealed record OptimizationChildResult(
    int ProcessId, int ExitCode, double WallMilliseconds, long? FootprintBytes, long? WorkingSetBytes,
    OptimizationWorkerResult? Result, string? Error);

internal static class OptimizationChildProcess
{
    public static async Task<OptimizationChildResult> RunAsync(
        OptimizationRequest request, int timeoutSeconds, CancellationToken cancellationToken)
    {
        var directory = Path.Combine(Path.GetTempPath(), "synapse-optimization-" + Guid.NewGuid().ToString("N"));
        _ = Directory.CreateDirectory(directory);
        try
        {
            var requestPath = Path.Combine(directory, "request.json");
            var outputPath = Path.Combine(directory, "output.json");
            await File.WriteAllTextAsync(requestPath, JsonSerializer.Serialize(request, OptimizationJsonContext.Default.OptimizationRequest),
                cancellationToken).ConfigureAwait(false);
            return await LaunchAsync(requestPath, outputPath, timeoutSeconds, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static async Task<OptimizationChildResult> LaunchAsync(
        string requestPath, string outputPath, int timeoutSeconds, CancellationToken cancellationToken)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(timeout.Token, cancellationToken);
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (var argument in new[] { Assembly.GetExecutingAssembly().Location, "optimization-worker", "--request", requestPath, "--output", outputPath })
        {
            start.ArgumentList.Add(argument);
        }

        var wall = Stopwatch.StartNew();
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Optimization worker did not start.");
        var output = process.StandardOutput.ReadToEndAsync(CancellationToken.None);
        var error = ReadErrorsAsync(process);
        await using var sampler = new ProcessMemorySampler(process);
        var timedOut = false;
        try
        {
            await process.WaitForExitAsync(stop.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
            timedOut = timeout.IsCancellationRequested && !cancellationToken.IsCancellationRequested;
        }

        var metrics = await sampler.CompleteAsync().ConfigureAwait(false);
        wall.Stop();
        _ = await output.ConfigureAwait(false);
        var (result, errors) = await ReadResultAsync(outputPath, await error.ConfigureAwait(false)).ConfigureAwait(false);
        var exit = cancellationToken.IsCancellationRequested ? 130 : timedOut ? 124 : process.ExitCode;
        if (exit == 0 && result is not { Status: "completed" })
        {
            exit = 1;
            errors += "Worker returned success without completed evidence.";
        }
        return new OptimizationChildResult(process.Id, exit, wall.Elapsed.TotalMilliseconds, metrics.PeakPhysicalFootprintBytes,
            metrics.MaximumObservedWorkingSetBytes, result,
            exit == 0 ? null : timedOut ? "Worker exceeded the explicit timeout; process tree terminated. " + errors : errors);
    }

    private static async Task<(OptimizationWorkerResult? Result, string Error)> ReadResultAsync(string outputPath, string errors)
    {
        try
        {
            var result = File.Exists(outputPath)
                ? JsonSerializer.Deserialize(await File.ReadAllTextAsync(outputPath, CancellationToken.None).ConfigureAwait(false),
                    OptimizationJsonContext.Default.OptimizationWorkerResult) : null;
            return (result, errors);
        }
        catch (Exception exception) when (exception is JsonException or IOException)
        {
            return (null, errors + "Worker evidence could not be read: " + exception);
        }
    }

    private static async Task<string> ReadErrorsAsync(Process process)
    {
        var error = new StringBuilder();
        while (await process.StandardError.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            await Console.Error.WriteLineAsync(line).ConfigureAwait(false);
            if (error.Length < 1024 * 1024)
            {
                _ = error.AppendLine(line);
            }
        }

        return error.ToString();
    }
}
