using System.Text;

internal static class HostedArtifactTable
{
    public static string Render(IReadOnlyList<HostedResultRow> rows, int complete, int expected,
        IReadOnlyList<string> missing, IReadOnlyList<string> invalid)
    {
        var report = new StringBuilder()
            .AppendLine("# Combined performance diagnostic")
            .AppendLine()
            .Append(complete).Append('/').Append(expected)
            .AppendLine(" expected raw artifacts complete. This is a diagnostic, not a winner verdict.")
            .AppendLine()
            .AppendLine("| Cohort | Runner | Scenario | Turn | Subject | Raw artifact | n | Output tok | TTFT ms | Reported decode/eval tok/s | Wall ms | Peak RSS MiB | Output state |")
            .AppendLine("|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|");

        foreach (var row in rows.OrderBy(row => row.CohortOrder)
            .ThenBy(row => row.Runner, StringComparer.Ordinal)
            .ThenBy(row => row.ScenarioOrder)
            .ThenBy(row => row.Turn)
            .ThenBy(row => row.Subject, StringComparer.Ordinal))
        {
            _ = report.Append('|').Append(Escape(row.Cohort))
                .Append('|').Append(Escape(row.Runner))
                .Append('|').Append(Escape(row.Scenario))
                .Append('|').Append(row.Turn)
                .Append('|').Append(Escape(row.Subject))
                .Append('|').Append(Escape(row.SourceArtifact))
                .Append('|').Append(row.Samples)
                .Append('|').Append(row.OutputTokens)
                .Append('|').Append(row.TtftMilliseconds)
                .Append('|').Append(row.DecodeTokensPerSecond)
                .Append('|').Append(row.WallMilliseconds)
                .Append('|').Append(row.PeakRssMiB)
                .Append('|').Append(Escape(row.OutputState)).AppendLine("|");
        }

        _ = report.AppendLine()
            .AppendLine("GGUF CPU wall time covers a fresh process and model load. MLX and Foundry wall time covers a request to a resident model; MLX uses Metal and different weights, and Foundry uses ONNX weights. Reported llama.cpp eval throughput is its native internal phase. Medians are calculated per column from measured rounds; warm-ups are excluded. `n/a` means the raw subject did not report that metric. Long-output quality is unreviewed; a token limit or empty final answer is not a passing quality result. Raw JSON artifacts remain the source of truth.");

        if (missing.Count > 0)
        {
            _ = report.AppendLine().Append("## Missing artifacts (").Append(missing.Count)
                .AppendLine(")").AppendLine();
            foreach (var item in missing.Order(StringComparer.Ordinal))
            {
                _ = report.Append("- `").Append(Escape(item)).AppendLine("`");
            }
        }

        if (invalid.Count > 0)
        {
            _ = report.AppendLine().Append("## Invalid artifacts (").Append(invalid.Count)
                .AppendLine(")").AppendLine();
            foreach (var item in invalid.Order(StringComparer.Ordinal))
            {
                _ = report.Append("- ").Append(Escape(item)).AppendLine();
            }
        }

        return report.ToString();
    }

    private static string Escape(string value) => value.Replace("|", "\\|", StringComparison.Ordinal)
        .Replace("\r", " ", StringComparison.Ordinal).Replace("\n", " ", StringComparison.Ordinal);
}

internal sealed record HostedResultRow(int CohortOrder, string Cohort, string Runner,
    int ScenarioOrder, string Scenario, int Turn, string Subject, string SourceArtifact, int Samples,
    string OutputTokens, string TtftMilliseconds, string DecodeTokensPerSecond,
    string WallMilliseconds, string PeakRssMiB, string OutputState,
    PerformanceModelDescriptor? Model = null);
