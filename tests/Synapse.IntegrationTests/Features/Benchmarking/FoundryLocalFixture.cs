using System.Diagnostics;
using System.Reflection;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

internal static class FoundryLocalFixture
{
    public const string AnchorAlias = "qwen2.5-0.5b";

    public static string ModelSetPath => Path.Combine(
        ReferenceBenchmarkFixture.FindRepositoryRoot(), "benchmarks", "model-sets", "foundry-local-families.json");

    public static string ScenarioPath(string name) => Path.Combine(
        ReferenceBenchmarkFixture.FindRepositoryRoot(), "benchmarks", "scenarios", name);

    public static string CacheDirectory
    {
        get
        {
            var configured = Environment.GetEnvironmentVariable("SYNAPSE_FOUNDRY_CACHE");
            return string.IsNullOrWhiteSpace(configured)
                ? Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), "artifacts", "foundry-local")
                : Path.GetFullPath(configured);
        }
    }

    public static async Task<SubjectProcessResult> RunAsync(params string[] arguments)
    {
        var assembly = FindBenchmarkAssembly();
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(assembly);
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start)
            ?? throw new InvalidOperationException("The Foundry Local benchmark process did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(10));
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync();
            throw new TimeoutException($"Foundry Local '{arguments[0]}' exceeded ten minutes.");
        }

        return new SubjectProcessResult(process.ExitCode, await output, await error);
    }

    private static string FindBenchmarkAssembly()
    {
        var configuration = typeof(FoundryLocalFixture).Assembly
            .GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration ?? "Release";
        var path = Path.Combine(ReferenceBenchmarkFixture.FindRepositoryRoot(), "experiments",
            "Synapse.FoundryLocalBenchmarks", "bin", configuration, "net10.0",
            "Synapse.FoundryLocalBenchmarks.dll");
        return File.Exists(path)
            ? path
            : throw new FileNotFoundException("The Foundry Local benchmark project was not built.", path);
    }
}
