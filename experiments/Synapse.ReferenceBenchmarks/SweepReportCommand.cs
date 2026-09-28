using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// <c>sweep-report</c>: rebuilds the three-axis summary from raw sweep samples. Several evidence files merge in order;
/// a later file's samples replace every earlier sample of the same (context, engine, KV) cell, which is how a corrected
/// rerun of one engine replaces its invalid rows without re-measuring the others.
/// </summary>
internal static class SweepReportCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index + 1 < args.Length; index += 2)
        {
            values[args[index]] = args[index + 1];
        }

        if (args.Length % 2 != 0 || !values.TryGetValue("--inputs", out var inputs) || !values.TryGetValue("--output", out var output))
        {
            await Console.Error.WriteLineAsync("Usage: sweep-report --inputs <a.json,b.json,...> --output <merged.json> [--reference <engine/kv>]")
                .ConfigureAwait(false);
            return 2;
        }

        var files = inputs.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var evidences = new List<SweepEvidence>();
        foreach (var file in files)
        {
            evidences.Add(JsonSerializer.Deserialize(await File.ReadAllTextAsync(file).ConfigureAwait(false),
                SweepJsonContext.Default.SweepEvidence) ?? throw new InvalidDataException($"{file} is empty."));
        }

        var samples = Merge(evidences);
        var first = evidences[0];
        var reference = values.GetValueOrDefault("--reference", first.Reference);
        var rows = SweepReport.Summarize(reference, first.Model, TextTokenizers.FromGguf(first.ModelPath), samples);
        var merged = first with
        {
            Kind = "context-sweep-diagnostic-merged",
            RecordedAt = DateTimeOffset.UtcNow,
            Reference = reference,
            Samples = samples,
            Summary = rows,
            SummaryMarkdown = SweepReport.Markdown(rows),
            Sources = [.. files.Select(Path.GetFileName).OfType<string>()],
        };
        await File.WriteAllTextAsync(output, JsonSerializer.Serialize(merged, SweepJsonContext.Default.SweepEvidence))
            .ConfigureAwait(false);
        Console.WriteLine(merged.SummaryMarkdown);
        return 0;
    }

    internal static List<SweepSample> Merge(IReadOnlyList<SweepEvidence> evidences)
    {
        var samples = new List<SweepSample>();
        foreach (var evidence in evidences)
        {
            var replaced = evidence.Samples.Select(sample => (sample.Context, sample.Subject, sample.KvCache)).ToHashSet();
            _ = samples.RemoveAll(sample => replaced.Contains((sample.Context, sample.Subject, sample.KvCache)));
            samples.AddRange(evidence.Samples);
        }

        return samples;
    }
}
