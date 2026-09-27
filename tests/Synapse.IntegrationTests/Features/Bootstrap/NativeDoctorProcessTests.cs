using System.Diagnostics;
using System.Text.Json;

namespace ManagedCode.Synapse.IntegrationTests.Features.Bootstrap;

public sealed class NativeDoctorProcessTests
{
    private const long OneGibibyte = 1L << 30;

    [Test]
    public async Task NativeDoctorReportsCpuThroughRealProcess()
    {
        var repositoryRoot = FindRepositoryRoot();
        var cargoExecutable = ResolveCargoExecutable();
        var startInfo = new ProcessStartInfo
        {
            FileName = cargoExecutable,
            WorkingDirectory = repositoryRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        PrependToolDirectoryToPath(startInfo, cargoExecutable);

        startInfo.ArgumentList.Add("run");
        startInfo.ArgumentList.Add("--manifest-path");
        startInfo.ArgumentList.Add("native/Cargo.toml");
        startInfo.ArgumentList.Add("--locked");
        startInfo.ArgumentList.Add("--quiet");
        startInfo.ArgumentList.Add("-p");
        startInfo.ArgumentList.Add("synapse-runtime");
        startInfo.ArgumentList.Add("--");
        startInfo.ArgumentList.Add("doctor");
        startInfo.ArgumentList.Add("--memory-budget-bytes");
        startInfo.ArgumentList.Add(OneGibibyte.ToString(System.Globalization.CultureInfo.InvariantCulture));

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Cargo process did not start.");
        var standardOutput = await process.StandardOutput.ReadToEndAsync();
        var standardError = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMinutes(2));

        await Assert.That(process.ExitCode).IsEqualTo(0)
            .Because(standardError);

        using var json = JsonDocument.Parse(standardOutput);
        await Assert.That(json.RootElement.GetProperty("supported").GetBoolean()).IsTrue();
        await Assert.That(json.RootElement.GetProperty("memoryBudgetBytes").GetInt64())
            .IsEqualTo(OneGibibyte);
        await Assert.That(json.RootElement.GetProperty("architecture").GetString())
            .IsNotNullOrEmpty();
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

    private static string ResolveCargoExecutable()
    {
        const string HomebrewCargo = "/opt/homebrew/opt/rustup/bin/cargo";
        return File.Exists(HomebrewCargo) ? HomebrewCargo : "cargo";
    }

    private static void PrependToolDirectoryToPath(
        ProcessStartInfo startInfo,
        string cargoExecutable)
    {
        var toolDirectory = Path.GetDirectoryName(cargoExecutable);
        if (string.IsNullOrWhiteSpace(toolDirectory))
        {
            return;
        }

        var existingPath = startInfo.Environment.TryGetValue("PATH", out var value)
            ? value
            : Environment.GetEnvironmentVariable("PATH");
        startInfo.Environment["PATH"] = string.IsNullOrWhiteSpace(existingPath)
            ? toolDirectory
            : $"{toolDirectory}{Path.PathSeparator}{existingPath}";
    }
}
