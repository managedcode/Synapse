using System.Text.Json;

internal static class TestResultsPublicationCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        if (!TryParse(args, out var results, out var runner, out var output))
        {
            Console.Error.WriteLine("Usage: test-report --results <trx-directory> " +
                "--runner <osx-arm64|linux-x64|win-x64> --output <new.json>");
            return 2;
        }

        try
        {
            var evidence = TestResultsPublicationReader.Read(results, runner);
            await PublishAsync(output, evidence).ConfigureAwait(false);
            Console.WriteLine($"Test results: {Path.GetFullPath(output)} ({evidence.Status}, {evidence.Total} tests)");
            return evidence.Status == "passed" ? 0 : 3;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ArgumentException)
        {
            Console.Error.WriteLine($"Cannot publish test results: {exception.Message}");
            return 1;
        }
    }

    private static bool TryParse(string[] args, out string results, out string runner, out string output)
    {
        results = runner = output = string.Empty;
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (index + 1 >= args.Length || args[index] is not ("--results" or "--runner" or "--output") ||
                string.IsNullOrWhiteSpace(args[index + 1]) || !values.TryAdd(args[index], args[index + 1]))
            {
                return false;
            }
        }

        return values.TryGetValue("--results", out results!) &&
            values.TryGetValue("--runner", out runner!) && runner is "osx-arm64" or "linux-x64" or "win-x64" &&
            values.TryGetValue("--output", out output!);
    }

    private static async Task PublishAsync(string output, TestResultsPublicationEvidence evidence)
    {
        var destination = Path.GetFullPath(output);
        var parent = Path.GetDirectoryName(destination)!;
        _ = Directory.CreateDirectory(parent);
        var temporary = Path.Combine(parent, $".{Path.GetFileName(destination)}.{Guid.NewGuid():N}.tmp");
        try
        {
            await using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                await JsonSerializer.SerializeAsync(stream, evidence,
                    TestResultsPublicationJsonContext.Default.TestResultsPublicationEvidence).ConfigureAwait(false);
                await stream.FlushAsync().ConfigureAwait(false);
            }

            File.Move(temporary, destination, overwrite: false);
        }
        finally
        {
            File.Delete(temporary);
        }
    }
}
