using System.Text.Json.Serialization;

internal sealed record TestResultsPublicationEvidence(
    int SchemaVersion,
    DateTimeOffset GeneratedAtUtc,
    string RuntimeIdentifier,
    string Scope,
    string Status,
    long Total,
    long Passed,
    long Failed,
    long Skipped,
    double? DurationSeconds,
    IReadOnlyList<string> Issues);

internal sealed record TrxReportSummary(
    string RunId,
    long Total,
    long Passed,
    long Failed,
    long Skipped,
    double? DurationSeconds,
    bool FailedRun);

[JsonSerializable(typeof(TestResultsPublicationEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
internal sealed partial class TestResultsPublicationJsonContext : JsonSerializerContext;
