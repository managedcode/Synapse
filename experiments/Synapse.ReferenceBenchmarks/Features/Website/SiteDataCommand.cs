using System.Text.Json;
using System.Text.Json.Nodes;

internal static class SiteDataCommand
{
    private static readonly string[] Runners = ["osx-arm64", "linux-x64", "win-x64"];

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length != 6)
        {
            Console.Error.WriteLine("Usage: site-data --artifacts <directory> --runs <runs.json> --output <new.json>");
            return 2;
        }

        try
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var index = 0; index < args.Length; index += 2)
            {
                if (args[index] is not ("--artifacts" or "--runs" or "--output") ||
                    string.IsNullOrWhiteSpace(args[index + 1]) || !values.TryAdd(args[index], args[index + 1]))
                {
                    return 2;
                }
            }

            var runs = await SitePublicationValidation.ReadAsync(values["--runs"]).ConfigureAwait(false);
            var verification = SourceRun(runs, "verification");
            var performance = SourceRun(runs, "performance");
            SitePublicationValidation.Run(verification);
            SitePublicationValidation.Run(performance);
            var data = new JsonObject
            {
                ["schema_version"] = 1,
                ["generated_at_utc"] = DateTimeOffset.UtcNow,
                ["verification"] = await VerificationAsync(values["--artifacts"], verification).ConfigureAwait(false),
                ["performance"] = await PerformanceAsync(values["--artifacts"], performance).ConfigureAwait(false),
            };
            await WriteAsync(values["--output"], data).ConfigureAwait(false);
            return 0;
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or JsonException or
            InvalidOperationException or ArgumentException or KeyNotFoundException)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static JsonObject? SourceRun(JsonObject runs, string name)
    {
        if (!runs.ContainsKey(name) || (runs[name] is not null && runs[name] is not JsonObject))
        {
            throw new InvalidDataException($"Malformed {name} source run metadata.");
        }

        return runs[name] as JsonObject;
    }

    private static async Task<JsonObject> VerificationAsync(string root, JsonObject? run)
    {
        var reports = new JsonArray();
        var missing = new JsonArray();
        foreach (var runner in Runners)
        {
            var artifact = $"test-results-{runner}";
            var path = Path.Combine(root, "verification", artifact, "test-results.json");
            try
            {
                if (run is null || !File.Exists(path))
                {
                    missing.Add(artifact);
                    continue;
                }

                var report = await SitePublicationValidation.ReadAsync(path).ConfigureAwait(false);
                SitePublicationValidation.Test(report, runner);
                reports.Add(report);
            }
            catch (Exception exception) when (exception is IOException or InvalidDataException or
                JsonException or InvalidOperationException)
            {
                missing.Add($"{artifact}: invalid report ({exception.Message})");
            }
        }

        return new JsonObject
        {
            ["run"] = run?.DeepClone(),
            ["reports"] = reports,
            ["missing_artifacts"] = missing,
        };
    }

    private static async Task<JsonObject> PerformanceAsync(string root, JsonObject? run)
    {
        JsonObject? report = null;
        var missing = new JsonArray();
        var path = Path.Combine(root, "performance", "performance-summary", "performance-results.json");
        try
        {
            if (run is not null && File.Exists(path))
            {
                report = await SitePublicationValidation.ReadAsync(path).ConfigureAwait(false);
                SitePublicationValidation.Performance(report);
                await PerformanceModelEnrichment.EnrichAsync(root, report, missing).ConfigureAwait(false);
            }
            else
            {
                missing.Add("performance-summary/performance-results.json");
            }
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or
            JsonException or InvalidOperationException)
        {
            report = null;
            missing.Add($"performance-summary: invalid report ({exception.Message})");
        }

        return new JsonObject
        {
            ["run"] = run?.DeepClone(),
            ["report"] = report,
            ["missing_artifacts"] = missing,
        };
    }

    private static async Task WriteAsync(string output, JsonObject data)
    {
        var destination = Path.GetFullPath(output);
        _ = Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var temporary = destination + $".{Guid.NewGuid():N}.tmp";
        try
        {
            await File.WriteAllTextAsync(temporary,
                data.ToJsonString(new JsonSerializerOptions { WriteIndented = true })).ConfigureAwait(false);
            File.Move(temporary, destination, overwrite: false);
        }
        finally
        {
            File.Delete(temporary);
        }
    }
}
