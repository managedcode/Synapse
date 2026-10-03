using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

internal static class ReferenceBenchmarkFixture
{
    public const string Prompt = "The capital of France is";
    public static readonly int[] PromptTokens = [785, 6722, 315, 9625, 374];
    private static readonly SemaphoreSlim CompilationGate = new(1, 1);
    private static string? _compiledModelPath;

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

    public static string GetModelPath()
    {
        var id = Environment.GetEnvironmentVariable("SYNAPSE_GGUF_MODEL_ID");
        var file = Environment.GetEnvironmentVariable("SYNAPSE_GGUF_MODEL_FILE");
        if (!string.IsNullOrWhiteSpace(id) && !string.IsNullOrWhiteSpace(file))
        {
            return Path.Combine(GetModelRoot(), id, file);
        }

        var catalog = ModelPackageCatalog.Load(
            Path.Combine(FindRepositoryRoot(), "models", "catalog.json"));
        var package = string.IsNullOrWhiteSpace(id) ? catalog.SelectSet("smoke").Single() : catalog.GetRequired(id);
        file = string.IsNullOrWhiteSpace(file)
            ? package.Files.Single(entry => entry.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).Path : file;
        return Path.Combine(GetModelRoot(), package.Id, file);
    }

    public static async Task<string> GetCompiledModelPathAsync(CancellationToken cancellationToken = default)
    {
        await CompilationGate.WaitAsync(cancellationToken);
        try
        {
            if (_compiledModelPath is not null && File.Exists(_compiledModelPath))
            {
                return _compiledModelPath;
            }

            var source = GetModelPath();
            await using var stream = File.OpenRead(source);
            var digest = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken));
            var destination = Path.Combine(FindRepositoryRoot(), "artifacts", "compiled-tests", digest + ".synapse");
            if (!File.Exists(destination))
            {
                try
                {
                    _ = await CompiledPackageCompiler.CompileAsync(source, destination, cancellationToken);
                }
                catch (IOException) when (File.Exists(destination))
                {
                    // A different test process may have atomically published the same content-addressed fixture.
                }
            }

            var info = CompiledPackageReader.Inspect(destination, cancellationToken);
            if (!string.Equals(info.SourceSha256, digest, StringComparison.Ordinal))
            {
                throw new InvalidDataException("Compiled test fixture does not match the pinned source content.");
            }

            _compiledModelPath = destination;
            return destination;
        }
        finally
        {
            CompilationGate.Release();
        }
    }

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

    public static string BenchmarkInputPath(string directory, string file) => Path.Combine(
        FindRepositoryRoot(), "experiments", "Synapse.ReferenceBenchmarks", "Features", "Benchmarking",
        directory, file);

    public static string RecordedBenchmarkPath(string file) => Path.Combine(
        FindRepositoryRoot(), "tests", "Synapse.IntegrationTests", "Features", "Benchmarking", "Fixtures", file);
}

internal sealed record SubjectProcessResult(int ExitCode, string StandardOutput, string StandardError);
