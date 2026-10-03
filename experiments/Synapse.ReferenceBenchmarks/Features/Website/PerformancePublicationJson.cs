using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

internal static class PerformancePublicationJson
{
    public static async Task WriteAsync(string path, IReadOnlyList<HostedResultRow> rows,
        int complete, int expected, IReadOnlyList<string> missing, IReadOnlyList<string> invalid,
        CancellationToken cancellationToken = default)
    {
        var destination = Path.GetFullPath(path);
        var temporary = destination + $".{Guid.NewGuid():N}.tmp";
        var snapshot = new PerformancePublicationSnapshot(1, DateTimeOffset.UtcNow, complete, expected,
            [.. missing.Order(StringComparer.Ordinal)], [.. invalid.Order(StringComparer.Ordinal)],
            [.. rows.OrderBy(row => row.CohortOrder)
                .ThenBy(row => row.Runner, StringComparer.Ordinal)
                .ThenBy(row => row.ScenarioOrder)
                .ThenBy(row => row.Turn)
                .ThenBy(row => row.Subject, StringComparer.Ordinal)
                .Select(Project)]);
        try
        {
            await using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write,
                FileShare.None, 4096, FileOptions.Asynchronous))
            {
                await JsonSerializer.SerializeAsync(stream, snapshot,
                    PerformancePublicationJsonContext.Default.PerformancePublicationSnapshot,
                    cancellationToken).ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, destination, overwrite: false);
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    internal static PerformancePublicationRow Project(HostedResultRow row) => new(
        row.Cohort, row.Runner, row.Scenario, row.Turn, row.Subject, row.SourceArtifact, row.Samples,
        row.OutputState, Metric(row.OutputTokens), Metric(row.TtftMilliseconds),
        Metric(row.DecodeTokensPerSecond), Metric(row.WallMilliseconds), Metric(row.PeakRssMiB),
        row.Subject == "llamacpp" ? "native_eval" : "reported_decode",
        row.CohortOrder == 0 ? "fresh_process" : "request", row.Model);

    private static double? Metric(string value)
    {
        if (value == "n/a")
        {
            return null;
        }

        return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture,
            out var number) && double.IsFinite(number)
            ? number : throw new InvalidDataException($"Invalid published metric '{value}'.");
    }
}

internal sealed record PerformancePublicationSnapshot(int SchemaVersion, DateTimeOffset GeneratedAtUtc,
    int CompleteArtifacts, int ExpectedArtifacts, string[] MissingArtifacts, string[] InvalidArtifacts,
    PerformancePublicationRow[] Rows);

internal sealed record PerformancePublicationRow(string Cohort, string Runner, string Scenario,
    int Turn, string Subject, string SourceArtifact, int Samples, string OutputState,
    double? OutputTokens, double? TtftMilliseconds, double? DecodeTokensPerSecond,
    double? WallMilliseconds, double? PeakRssMib, string DecodeMetricScope, string WallMetricScope,
    PerformanceModelDescriptor? Model);

[JsonSerializable(typeof(PerformancePublicationSnapshot))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    WriteIndented = true)]
internal sealed partial class PerformancePublicationJsonContext : JsonSerializerContext;
