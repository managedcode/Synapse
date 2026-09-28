using System.Diagnostics;
using System.Text.Json;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

internal static class ReferenceBenchmarkFixture
{
    public const string Prompt = "The capital of France is";
    public static readonly int[] PromptTokens = [785, 6722, 315, 9625, 374];

    public static async Task<JsonDocument> RunSubjectAsync(
        string subject,
        params string[] additionalArguments)
    {
        var result = await RunSubjectProcessAsync(subject, additionalArguments);
        await Assert.That(result.ExitCode).IsEqualTo(0).Because(result.StandardError);
        return JsonDocument.Parse(result.StandardOutput);
    }

    public static async Task<SubjectProcessResult> RunSubjectProcessAsync(
        string subject,
        params string[] additionalArguments)
    {
        var benchmarkAssembly = Path.Combine(AppContext.BaseDirectory, "Synapse.ReferenceBenchmarks.dll");
        if (!File.Exists(benchmarkAssembly))
        {
            throw new FileNotFoundException("The reference benchmark process was not built.", benchmarkAssembly);
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        AddArguments(startInfo, benchmarkAssembly, subject, additionalArguments);
        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("The reference benchmark process did not start.");
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            await process.WaitForExitAsync();
            throw new TimeoutException($"The {subject} reference process exceeded five minutes.");
        }

        return new SubjectProcessResult(process.ExitCode, await outputTask, await errorTask);
    }

    public static string GetModelPath() => Path.Combine(
        GetModelRoot(),
        "qwen2.5-0.5b-instruct-q8_0",
        "qwen2.5-0.5b-instruct-q8_0.gguf");

    public static string RequireEnvironmentFile(string variableName)
    {
        var path = Environment.GetEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new InvalidOperationException(
                $"{variableName} must point to a built real executable; reference smoke tests do not skip.");
        }

        return path;
    }

    private static void AddArguments(
        ProcessStartInfo startInfo,
        string benchmarkAssembly,
        string subject,
        string[] additionalArguments)
    {
        string[] requiredArguments =
        [
            benchmarkAssembly,
            subject,
            "--model", GetModelPath(),
            "--prompt", Prompt,
            "--max-tokens", "8",
            "--threads", "8",
        ];
        foreach (var argument in requiredArguments.Concat(additionalArguments))
        {
            startInfo.ArgumentList.Add(argument);
        }
    }

    private static string GetModelRoot()
    {
        var configured = Environment.GetEnvironmentVariable("SYNAPSE_MODEL_ROOT");
        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(FindRepositoryRoot(), "artifacts", "models")
            : Path.GetFullPath(configured);
    }

    public static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Synapse.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the Synapse repository root.");
    }
}

internal sealed record SubjectProcessResult(int ExitCode, string StandardOutput, string StandardError);
