using System.Text.Json;
using System.Text.Json.Nodes;

internal static class SitePublicationValidation
{
    public static async Task<JsonObject> ReadAsync(string path)
    {
        var file = new FileInfo(path);
        if (file.Length > 16 * 1024 * 1024 || file.LinkTarget is not null)
        {
            throw new InvalidDataException("Report exceeds the size bound or is a symbolic link.");
        }

        await using var stream = file.OpenRead();
        return await JsonNode.ParseAsync(stream, documentOptions: new JsonDocumentOptions { MaxDepth = 32 })
            .ConfigureAwait(false) as JsonObject ?? throw new InvalidDataException("Expected a JSON object.");
    }

    public static void Run(JsonObject? run)
    {
        if (run is null)
        {
            return;
        }

        var id = Count(run, "id");
        var sha = Text(run, "head_sha");
        Require(id > 0 && Count(run, "run_number") > 0 && Count(run, "run_attempt") > 0);
        Require(sha.Length == 40 && sha.All(character => character is (>= '0' and <= '9') or (>= 'a' and <= 'f')));
        Require(Text(run, "html_url") == $"https://github.com/managedcode/Synapse/actions/runs/{id}");
        Require(Text(run, "repository") == "managedcode/Synapse" && Text(run, "head_branch") == "main");
        Require(Text(run, "event") is "push" or "workflow_dispatch");
        _ = Text(run, "conclusion");
        Date(run, "updated_at");
    }

    public static void Test(JsonObject report, string runner)
    {
        Schema(report);
        Require(Text(report, "runtime_identifier") == runner);
        Require(Text(report, "scope") == ".NET functional tests");
        var status = Text(report, "status");
        Require(status is "passed" or "failed" or "not_run" or "invalid");
        var total = Count(report, "total");
        var passed = Count(report, "passed");
        var failed = Count(report, "failed");
        var skipped = Count(report, "skipped");
        Require(passed + failed + skipped == total);
        Require(status != "passed" || (passed > 0 && failed == 0));
        Metric(report, "duration_seconds");
        Strings(report, "issues");
    }

    public static void Performance(JsonObject report)
    {
        Schema(report);
        Require(Count(report, "complete_artifacts") <= Count(report, "expected_artifacts"));
        Strings(report, "missing_artifacts");
        Strings(report, "invalid_artifacts");
        var rows = report["rows"] as JsonArray ?? throw new InvalidDataException("No benchmark rows.");
        Require(rows.Count <= 10000);
        foreach (var node in rows)
        {
            var row = node as JsonObject ?? throw new InvalidDataException("Invalid benchmark row.");
            foreach (var field in new[] { "cohort", "runner", "scenario", "subject", "source_artifact", "output_state" })
            {
                _ = Text(row, field);
            }

            Require(Count(row, "samples") > 0 && Count(row, "turn") > 0);
            Require(Text(row, "decode_metric_scope") is "native_eval" or "reported_decode");
            Require(Text(row, "wall_metric_scope") is "fresh_process" or "request");
            if (row["model"] is JsonObject model)
            {
                foreach (var field in new[] { "family", "label", "weights_sha256", "scenario_sha256", "execution", "hardware" })
                {
                    _ = Text(model, field);
                }
            }
            else
            {
                Require(row["model"] is null);
            }
            foreach (var field in new[] { "output_tokens", "ttft_milliseconds", "decode_tokens_per_second",
                "wall_milliseconds", "peak_rss_mib" })
            {
                Metric(row, field);
            }
        }
    }

    private static void Schema(JsonObject value)
    {
        Require(Count(value, "schema_version") == 1);
        Date(value, "generated_at_utc");
    }

    private static void Date(JsonObject value, string name) =>
        Require(DateTimeOffset.TryParse(Text(value, name), System.Globalization.CultureInfo.InvariantCulture,
            System.Globalization.DateTimeStyles.RoundtripKind, out _));

    private static long Count(JsonObject value, string name)
    {
        var count = value[name]?.GetValue<long>() ?? throw new InvalidDataException($"Missing {name}.");
        Require(count is >= 0 and <= 9007199254740991);
        return count;
    }

    private static string Text(JsonObject value, string name)
    {
        var text = value[name]?.GetValue<string>() ?? throw new InvalidDataException($"Missing {name}.");
        Require(!string.IsNullOrWhiteSpace(text) && text.Length <= 2048);
        return text;
    }

    private static void Metric(JsonObject value, string name)
    {
        Require(value.ContainsKey(name));
        if (value[name] is { } node)
        {
            var number = node.GetValue<double>();
            Require(double.IsFinite(number) && number >= 0);
        }
    }

    private static void Strings(JsonObject value, string name)
    {
        var array = value[name] as JsonArray ?? throw new InvalidDataException($"Missing {name}.");
        Require(array.Count <= 10000);
        foreach (var node in array)
        {
            var text = node?.GetValue<string>();
            Require(text is { Length: > 0 and <= 2048 });
        }
    }

    private static void Require(bool condition)
    {
        if (!condition)
        {
            throw new InvalidDataException("Invalid publication schema or provenance.");
        }
    }
}
