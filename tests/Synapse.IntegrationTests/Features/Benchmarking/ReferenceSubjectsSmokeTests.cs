using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel]
public sealed class ReferenceSubjectsSmokeTests
{
    private const string ExpectedModelSha256 =
        "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
    private const string Prompt = "The capital of France is";
    private static readonly int[] PromptTokens = [785, 6722, 315, 9625, 374];

    [Test]
    public async Task SharedQwenFixtureMatchesPinnedDigest()
    {
        var modelPath = GetModelPath();
        await using var stream = File.OpenRead(modelPath);
        var digest = await SHA256.HashDataAsync(stream);

        await Assert.That(Convert.ToHexStringLower(digest)).IsEqualTo(ExpectedModelSha256);
    }

    [Test]
    public async Task DotLlmGeneratesTokensFromSharedQwenFixture()
    {
        var executable = RequireEnvironmentFile("SYNAPSE_DOTLLM_EXECUTABLE");
        var version = Environment.GetEnvironmentVariable("SYNAPSE_DOTLLM_VERSION")
            ?? "d88040451d7db56e5dfef9d5754ad0955b0f7fe5";
        using var result = await RunSubjectAsync(
            "dotllm",
            "--subject-executable", executable,
            "--subject-version", version);

        await AssertSuccessfulGenerationAsync(result, "dotllm");
    }

    [Test]
    public async Task LlamaSharpGeneratesTokensFromSharedQwenFixture()
    {
        using var result = await RunSubjectAsync("llamasharp", "--backend", "cpu");

        await AssertSuccessfulGenerationAsync(result, "llamasharp");
    }

    [Test]
    public async Task SynapseManagedQwen2MatchesReferenceFirstToken()
    {
        using var model = ModelLoader.Load(GetModelPath(), contextSize: 512);

        var graphVerification = ModelGraphVerifier.Verify(model.Graph);
        var result = model.Generate(PromptTokens, maximumNewTokens: 1);

        await Assert.That(graphVerification.IsValid).IsTrue();
        await Assert.That(model.Architecture).IsEqualTo("qwen2");
        await Assert.That(model.Graph.Regions.Count).IsEqualTo(26);
        await Assert.That(model.Graph.Regions.All(region =>
            region.Eligibility is AlwaysRequiredEligibility)).IsTrue();
        await Assert.That(result.GeneratedTokens).IsEquivalentTo([12095]);
        await Assert.That(result.Elapsed).IsGreaterThan(TimeSpan.Zero);
    }

    private static async Task<JsonDocument> RunSubjectAsync(
        string subject,
        params string[] additionalArguments)
    {
        var benchmarkAssembly = Path.Combine(
            AppContext.BaseDirectory,
            "Synapse.ReferenceBenchmarks.dll");
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
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(5));
        var output = await outputTask;
        var error = await errorTask;

        await Assert.That(process.ExitCode).IsEqualTo(0)
            .Because(error);
        return JsonDocument.Parse(output);
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

    private static async Task AssertSuccessfulGenerationAsync(
        JsonDocument result,
        string expectedSubject)
    {
        var root = result.RootElement;
        await Assert.That(root.GetProperty("subject").GetString()).IsEqualTo(expectedSubject);
        await Assert.That(root.GetProperty("generated_tokens").GetInt32()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("text").GetString()).StartsWith(" Paris");
        await Assert.That(root.GetProperty("total_generation_milliseconds").GetDouble())
            .IsGreaterThan(0);
    }

    private static string GetModelPath() => Path.Combine(
            GetModelRoot(),
            "qwen2.5-0.5b-instruct-q8_0",
            "qwen2.5-0.5b-instruct-q8_0.gguf");

    private static string GetModelRoot()
    {
        var configured = Environment.GetEnvironmentVariable("SYNAPSE_MODEL_ROOT");
        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(FindRepositoryRoot(), "artifacts", "models")
            : Path.GetFullPath(configured);
    }

    private static string RequireEnvironmentFile(string variableName)
    {
        var path = Environment.GetEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new InvalidOperationException(
                $"{variableName} must point to a built real executable; reference smoke tests do not skip.");
        }

        return path;
    }

    private static string FindRepositoryRoot()
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
