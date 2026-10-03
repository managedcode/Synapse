using System.Globalization;
using System.Xml;
using System.Xml.Linq;

internal static class TestResultsPublicationReader
{
    private const long MaximumReportBytes = 64 * 1024 * 1024;
    private const int MaximumReports = 128;
    private static readonly XNamespace TrxNamespace = "http://microsoft.com/schemas/VisualStudio/TeamTest/2010";
    private static readonly string[] FailedCounters =
        ["failed", "error", "timeout", "aborted", "passedButRunAborted", "disconnected", "warning"];
    private static readonly string[] SkippedCounters = ["notExecuted", "notRunnable", "inconclusive"];

    public static TestResultsPublicationEvidence Read(string directory, string runner)
    {
        var issues = new List<string>();
        var reports = new List<TrxReportSummary>();
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            foreach (var path in ReportPaths(directory))
            {
                try
                {
                    var report = ReadReport(path);
                    if (!identifiers.Add(report.RunId))
                    {
                        throw new InvalidDataException("Duplicate TRX run identity.");
                    }

                    reports.Add(report);
                }
                catch (Exception exception) when (exception is IOException or InvalidDataException or XmlException or UnauthorizedAccessException)
                {
                    issues.Add($"{Path.GetFileName(path)}: {exception.Message}");
                }
            }
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            issues.Add($"TRX discovery failed: {exception.Message}");
        }

        var total = reports.Sum(report => report.Total);
        var passed = reports.Sum(report => report.Passed);
        var failed = reports.Sum(report => report.Failed);
        var skipped = reports.Sum(report => report.Skipped);
        var invalid = issues.Count != 0;
        if (reports.Count == 0 || total == 0 || passed + failed == 0)
        {
            issues.Add("No executed tests were reported; a required suite cannot pass with zero tests.");
        }
        else if (failed != 0 || reports.Exists(report => report.FailedRun))
        {
            issues.Add("The test run reported failures or an unsuccessful run outcome.");
        }

        var status = invalid ? "invalid" : total == 0 || passed + failed == 0 ? "not_run" :
            issues.Count != 0 ? "failed" : "passed";
        double? duration = reports.Count != 0 && reports.All(report => report.DurationSeconds.HasValue)
            ? reports.Sum(report => report.DurationSeconds!.Value) : null;
        return new TestResultsPublicationEvidence(1, DateTimeOffset.UtcNow, runner, ".NET functional tests",
            status, total, passed, failed, skipped, duration, issues);
    }

    private static string[] ReportPaths(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return [];
        }

        var options = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MaxRecursionDepth = 8,
            IgnoreInaccessible = false,
            AttributesToSkip = FileAttributes.ReparsePoint,
        };
        var paths = Directory.EnumerateFiles(directory, "*.trx", options).Take(MaximumReports + 1).ToArray();
        if (paths.Length > MaximumReports)
        {
            throw new InvalidDataException($"More than {MaximumReports} TRX reports exceeds the publication limit.");
        }

        return [.. paths.Order(StringComparer.Ordinal)];
    }

    private static TrxReportSummary ReadReport(string path)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length > MaximumReportBytes)
        {
            throw new InvalidDataException("TRX report exceeds the 64 MiB publication limit.");
        }

        using var reader = XmlReader.Create(stream, new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit,
            XmlResolver = null,
            MaxCharactersInDocument = MaximumReportBytes,
        });
        var document = XDocument.Load(reader);
        var root = document.Root;
        if (root?.Name != TrxNamespace + "TestRun" ||
            !Guid.TryParse((string?)root.Attribute("id"), out var runId))
        {
            throw new InvalidDataException("Expected a TRX TestRun with a valid run identity.");
        }

        var summary = root.Element(TrxNamespace + "ResultSummary") ??
            throw new InvalidDataException("TRX ResultSummary is missing.");
        var counters = summary.Element(TrxNamespace + "Counters") ??
            throw new InvalidDataException("TRX Counters is missing.");
        var total = Counter(counters, "total", required: true);
        var passed = Counter(counters, "passed", required: true);
        var failed = FailedCounters.Sum(name => Counter(counters, name, required: name == "failed"));
        var skipped = SkippedCounters.Sum(name => Counter(counters, name));
        if (total != passed + failed + skipped || Counter(counters, "inProgress") != 0 ||
            Counter(counters, "pending") != 0 || Counter(counters, "completed") != 0)
        {
            throw new InvalidDataException("TRX counters are inconsistent or contain incomplete tests.");
        }

        ValidateResults(root, total, passed, failed, skipped);
        var outcome = (string?)summary.Attribute("outcome");
        var failedRun = outcome is "Failed" or "Error" or "Aborted" or "Timeout" or "Disconnected";
        if (!failedRun && outcome is not ("Completed" or "Passed"))
        {
            throw new InvalidDataException("TRX run outcome is missing or unsupported.");
        }

        return new TrxReportSummary(runId.ToString("D"), total, passed, failed, skipped, Duration(root), failedRun);
    }

    private static void ValidateResults(XElement root, long total, long passed, long failed, long skipped)
    {
        var records = root.Element(TrxNamespace + "Results")?.Elements(TrxNamespace + "UnitTestResult").ToArray() ?? [];
        if (records.LongLength != total)
        {
            throw new InvalidDataException("TRX test records do not match the total counter.");
        }

        var actual = new long[3];
        var executions = new HashSet<string>(StringComparer.Ordinal);
        foreach (var record in records)
        {
            var execution = (string?)record.Attribute("executionId");
            if (!Guid.TryParse(execution, out _) || !executions.Add(execution))
            {
                throw new InvalidDataException("TRX test execution identity is missing or duplicated.");
            }

            var category = (string?)record.Attribute("outcome") switch
            {
                "Passed" => 0,
                "Failed" or "Error" or "Timeout" or "Aborted" or "PassedButRunAborted" or "Disconnected" or "Warning" => 1,
                "NotExecuted" or "NotRunnable" or "Inconclusive" => 2,
                _ => throw new InvalidDataException("TRX test outcome is missing or unsupported."),
            };
            actual[category]++;
        }

        if (actual[0] != passed || actual[1] != failed || actual[2] != skipped)
        {
            throw new InvalidDataException("TRX test outcomes do not reconcile with the summary counters.");
        }
    }

    private static long Counter(XElement counters, string name, bool required = false)
    {
        var value = (string?)counters.Attribute(name);
        if (value is null && !required)
        {
            return 0;
        }

        if (!long.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var result) ||
            result is < 0 or > 1_000_000)
        {
            throw new InvalidDataException($"Invalid TRX counter '{name}'.");
        }

        return result;
    }

    private static double? Duration(XElement root)
    {
        var times = root.Element(TrxNamespace + "Times");
        if (times is null)
        {
            return null;
        }

        if (!DateTimeOffset.TryParse((string?)times.Attribute("start"), CultureInfo.InvariantCulture,
                DateTimeStyles.RoundtripKind, out var start) ||
            !DateTimeOffset.TryParse((string?)times.Attribute("finish"), CultureInfo.InvariantCulture,
                DateTimeStyles.RoundtripKind, out var finish) || finish < start)
        {
            throw new InvalidDataException("TRX duration timestamps are invalid.");
        }

        return (finish - start).TotalSeconds;
    }
}
