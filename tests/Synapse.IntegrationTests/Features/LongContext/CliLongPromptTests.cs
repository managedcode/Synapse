using System.Diagnostics;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

[NotInParallel]
public sealed class CliLongPromptTests
{
    [Test]
    public async Task CliTokensFileMatchesInlineTokens()
    {
        var path = WriteTokensFile(string.Join('\n', ReferenceBenchmarkFixture.PromptTokens) + "\n");
        try
        {
            var (exitCode, output, error) = await RunCliAsync("--tokens-file", path, "--rope-scaling", "yarn:4:32768");

            await Assert.That(exitCode).IsEqualTo(0).Because(error);
            using var json = JsonDocument.Parse(output);
            var root = json.RootElement;
            await Assert.That(root.GetProperty("prompt_token_count").GetInt32()).IsEqualTo(5);
            await Assert.That(root.GetProperty("context_size").GetInt32()).IsEqualTo(64);
            await Assert.That(root.GetProperty("rope_scaling").GetString()).IsEqualTo("yarn4");
            await Assert.That(root.GetProperty("subject").GetString()).EndsWith("+yarn4");
            await Assert.That(error).Contains("prefill 5/5");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task CliRejectsMalformedTokensFile()
    {
        var malformed = WriteTokensFile("785, 6722\nParis\n");
        var inline = WriteTokensFile("785");
        try
        {
            var (badExit, _, badError) = await RunCliAsync("--tokens-file", malformed);
            var (bothExit, _, _) = await RunCliAsync("--tokens-file", inline, "--tokens", "785");
            var (missingExit, _, _) = await RunCliAsync("--tokens-file", malformed + ".missing");
            var (scalingExit, _, _) = await RunCliAsync("--tokens", "785", "--rope-scaling", "yarn:1:32768");

            await Assert.That(badExit).IsEqualTo(2);
            await Assert.That(badError).Contains("--tokens-file <path>");
            await Assert.That(bothExit).IsEqualTo(2);
            await Assert.That(missingExit).IsEqualTo(2);
            await Assert.That(scalingExit).IsEqualTo(2);
        }
        finally
        {
            File.Delete(malformed);
            File.Delete(inline);
        }
    }

    private static string WriteTokensFile(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-tokens-{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, content);
        return path;
    }

    private static async Task<(int ExitCode, string Output, string Error)> RunCliAsync(params string[] extra)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        string[] arguments =
        [
            Path.Combine(AppContext.BaseDirectory, "synapse.dll"),
            "generate",
            "--model", await ReferenceBenchmarkFixture.GetCompiledModelPathAsync(),
            "--max-tokens", "2",
            "--context-size", "64",
            "--threads", "2",
            .. extra,
        ];
        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("The Synapse CLI process did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(2));
        return (process.ExitCode, await output, await error);
    }
}
