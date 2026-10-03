using System.Text;
using System.Text.Json;

internal static class HostedArtifactReportCommand
{
    private static readonly string[] CpuRunners = ["osx-arm64", "linux-x64", "win-x64"];

    public static async Task<int> RunAsync(string[] args)
    {
        if (!TryParse(args, out var options))
        {
            Console.Error.WriteLine("Usage: aggregate --artifacts <directory> --model-set <set.json> " +
                "--output <new.md> [--summary <github-step-summary>] [--json <new.json>]");
            return 2;
        }

        try
        {
            var expected = await ExpectedAsync(options.ModelSet).ConfigureAwait(false);
            var rows = new List<HostedResultRow>();
            var missing = new List<string>();
            var invalid = new List<string>();
            if (Directory.Exists(options.Artifacts))
            {
                var names = expected.Select(artifact => artifact.Name).ToHashSet(StringComparer.Ordinal);
                foreach (var directory in Directory.EnumerateDirectories(options.Artifacts))
                {
                    var name = Path.GetFileName(directory);
                    if (name != "performance-summary" && !names.Contains(name))
                    {
                        invalid.Add($"unexpected artifact directory: {name}");
                    }
                }
            }

            var complete = 0;
            foreach (var artifact in expected)
            {
                if (await ReadArtifactAsync(options.Artifacts, artifact, rows, missing, invalid)
                    .ConfigureAwait(false))
                {
                    complete++;
                }
            }

            var report = HostedArtifactTable.Render(rows, complete, expected.Count, missing, invalid);
            await using (var output = new FileStream(options.Output, FileMode.CreateNew,
                FileAccess.Write, FileShare.None))
            await using (var writer = new StreamWriter(output, new UTF8Encoding(false)))
            {
                await writer.WriteAsync(report).ConfigureAwait(false);
            }

            if (options.Summary is not null)
            {
                await File.AppendAllTextAsync(options.Summary, report).ConfigureAwait(false);
            }

            if (options.Json is not null)
            {
                await PerformancePublicationJson.WriteAsync(options.Json, rows, complete,
                    expected.Count, missing, invalid).ConfigureAwait(false);
            }

            Console.WriteLine($"Combined report: {Path.GetFullPath(options.Output)} " +
                $"({complete}/{expected.Count} artifacts)");
            return missing.Count == 0 && invalid.Count == 0 ? 0 : 3;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static async Task<bool> ReadArtifactAsync(string root, HostedArtifactSpec artifact,
        List<HostedResultRow> rows, List<string> missing, List<string> invalid)
    {
        var directory = Path.Combine(root, artifact.Name);
        if (!Directory.Exists(directory))
        {
            missing.Add(artifact.Name);
            return false;
        }

        var complete = true;
        foreach (var file in artifact.Files)
        {
            var path = Path.Combine(directory, file);
            if (!File.Exists(path))
            {
                missing.Add($"{artifact.Name}/{file}");
                complete = false;
                continue;
            }

            try
            {
                await using var stream = File.OpenRead(path);
                using var document = await JsonDocument.ParseAsync(stream).ConfigureAwait(false);
                rows.AddRange(HostedEvidenceReader.Read(document.RootElement, artifact, file));
            }
            catch (Exception exception) when (exception is IOException or JsonException or
                InvalidDataException or KeyNotFoundException or InvalidOperationException)
            {
                invalid.Add($"{artifact.Name}/{file}: {exception.Message}");
                complete = false;
            }
        }

        return complete;
    }

    private static async Task<List<HostedArtifactSpec>> ExpectedAsync(string modelSetPath)
    {
        await using var stream = File.OpenRead(modelSetPath);
        using var document = await JsonDocument.ParseAsync(stream).ConfigureAwait(false);
        var root = document.RootElement;
        var setId = root.GetProperty("id").GetString()
            ?? throw new InvalidDataException("The model set has no id.");
        var expected = new List<HostedArtifactSpec>();
        foreach (var runner in CpuRunners)
        {
            expected.Add(new HostedArtifactSpec($"performance-{runner}", "smoke",
                ["synapse-benchmark.json"]));
            expected.Add(new HostedArtifactSpec($"performance-long-{runner}", "cpu",
                ["synapse-single.json", "synapse-dialogue.json"]));
        }

        expected.Add(new HostedArtifactSpec("performance-mlx-osx-arm64", "mlx",
            ["mlx-single.json", "mlx-dialogue.json"]));
        foreach (var runner in root.GetProperty("runners").EnumerateArray())
        {
            foreach (var model in root.GetProperty("models").EnumerateArray())
            {
                var cpu = model.GetProperty("variants").EnumerateArray()
                    .Single(variant => variant.GetProperty("device").GetString() == "cpu");
                if ((long)cpu.GetProperty("fileSizeMb").GetInt32() * 2 >
                    runner.GetProperty("memoryMb").GetInt32())
                {
                    continue;
                }

                var runnerId = runner.GetProperty("id").GetString();
                var alias = model.GetProperty("alias").GetString();
                expected.Add(new HostedArtifactSpec($"foundry-local-{runnerId}-{alias}", "foundry",
                    ["foundry-single.json", "foundry-dialogue.json"], alias, setId));
            }
        }

        return expected;
    }

    private static bool TryParse(string[] args, out HostedReportOptions options)
    {
        options = new HostedReportOptions(string.Empty, string.Empty, string.Empty, null, null);
        if (args.Length is not (6 or 8 or 10) || args.Length % 2 != 0)
        {
            return false;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (args[index] is not ("--artifacts" or "--model-set" or "--output" or "--summary" or "--json") ||
                string.IsNullOrWhiteSpace(args[index + 1]) || !values.TryAdd(args[index], args[index + 1]))
            {
                return false;
            }
        }

        if (!values.TryGetValue("--artifacts", out var artifacts) ||
            !values.TryGetValue("--model-set", out var modelSet) ||
            !values.TryGetValue("--output", out var output))
        {
            return false;
        }

        options = new HostedReportOptions(artifacts, modelSet, output,
            values.GetValueOrDefault("--summary"), values.GetValueOrDefault("--json"));
        return true;
    }
}

internal sealed record HostedReportOptions(string Artifacts, string ModelSet, string Output,
    string? Summary, string? Json);

internal sealed record HostedArtifactSpec(string Name, string Kind, string[] Files,
    string? Alias = null, string? ModelSetId = null);
