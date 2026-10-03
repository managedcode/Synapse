using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

internal static class FoundryLocalFixture
{
    public static string AnchorAlias
    {
        get
        {
            var configured = Environment.GetEnvironmentVariable("SYNAPSE_FOUNDRY_ANCHOR");
            if (!string.IsNullOrWhiteSpace(configured))
            {
                return configured;
            }

            using var definition = JsonDocument.Parse(File.ReadAllText(ModelSetPath));
            return definition.RootElement.GetProperty("models")[0].GetProperty("alias").GetString()
                ?? throw new InvalidDataException("The local model set has no default anchor alias.");
        }
    }

    public static string ModelSetPath => Environment.GetEnvironmentVariable("SYNAPSE_FOUNDRY_MODEL_SET") is { Length: > 0 } configured
        ? Path.GetFullPath(configured, ReferenceBenchmarkFixture.FindRepositoryRoot())
        : ReferenceBenchmarkFixture.BenchmarkInputPath("ModelSets", "foundry-local-families.json");

    public static string ScenarioPath(string name) => ReferenceBenchmarkFixture.BenchmarkInputPath("Scenarios", name);

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
