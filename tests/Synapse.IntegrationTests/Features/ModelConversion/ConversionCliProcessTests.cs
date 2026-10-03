using System.Diagnostics;
using System.Text.Json;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

[NotInParallel]
public sealed class ConversionCliProcessTests
{
    [Test]
    public async Task ConvertsInspectsAndRunsNativeGraphInRealProcesses()
    {
        using var fixture = new NativeConversionFixture();
        var (convertExit, convertOutput, convertError) = await CliAsync("model", "convert", "--source", fixture.Source,
            "--graph", fixture.Graph, "--output", fixture.Destination);
        await Assert.That(convertExit).IsEqualTo(0).Because(convertError);
        using var info = JsonDocument.Parse(convertOutput);
        await Assert.That(info.RootElement.GetProperty("source_format").GetString()).IsEqualTo("safetensors");
        File.Delete(fixture.Source);
        File.Delete(fixture.Graph);
        var (inspectExit, _, inspectError) = await CliAsync("model", "inspect", "--model", fixture.Destination);
        await Assert.That(inspectExit).IsEqualTo(0).Because(inspectError);
        var input = Path.ChangeExtension(fixture.Source, ".inputs.json");
        File.WriteAllText(input, /*lang=json,strict*/ """{"x":{"shape":[2,2],"data":[1,2,3,4]}}""");
        var (runExit, runOutput, runError) = await CliAsync("model", "run", "--model", fixture.Destination, "--inputs", input);
        await Assert.That(runExit).IsEqualTo(0).Because(runError);
        using var output = JsonDocument.Parse(runOutput);
        await Assert.That(output.RootElement.GetProperty("y").EnumerateArray().Select(value => value.GetSingle()).ToArray())
            .IsEquivalentTo([5f, 11f, 11f, 25f]);
    }

    [Test]
    public async Task GenerationFromSourceShowsExplicitConversionCommand()
    {
        using var fixture = new NativeConversionFixture();
        var (exit, _, error) = await CliAsync("generate", "--model", fixture.Source, "--tokens", "1", "--max-tokens", "1");
        await Assert.That(exit).IsEqualTo(1);
        await Assert.That(error).Contains("synapse model convert");
    }

    private static async Task<(int Exit, string Output, string Error)> CliAsync(params string[] arguments)
    {
        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "synapse.dll"));
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("CLI child process could not start.");
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var output = process.StandardOutput.ReadToEndAsync(cancellation.Token);
        var error = process.StandardError.ReadToEndAsync(cancellation.Token);
        try
        {
            await process.WaitForExitAsync(cancellation.Token);
            return (process.ExitCode, await output, await error);
        }
        finally
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
    }
}
