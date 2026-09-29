using System.Globalization;
using System.Text;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// One sweep cell: medians of the successful measured (non-warm-up) samples on the three axes. Failed runs count toward
/// <see cref="MeasuredRuns"/> and <see cref="FailedRuns"/>, so a crash never hides behind a clean-looking cell.
/// </summary>
internal sealed record SweepRow(
    int Context,
    string Subject,
    string KvCache,
    int? PromptTokens,
    bool? TokenIdsIdentical,
    double WeightsMebibytes,
    double? KvCacheMebibytes,
    double? PeakFootprintMebibytes,
    double? PeakResidentMebibytes,
    double? TimeToFirstTokenMilliseconds,
    double? GenerationMilliseconds,
    double? DecodeTokensPerSecond,
    int GeneratedTokens,
    int CorrectRuns,
    int MeasuredRuns,
    int FailedRuns,
    int? TokensMatchingReference);

internal static class SweepReport
{
    private const double Mebibyte = 1024 * 1024;

    public static IReadOnlyList<SweepRow> Summarize(
        string referenceCell,
        SweepModel model,
        ITextTokenizer tokenizer,
        IReadOnlyList<SweepSample> samples)
    {
        var rows = new List<SweepRow>();
        foreach (var cell in samples.Where(sample => !sample.Warmup)
            .GroupBy(sample => (sample.Context, sample.Subject, sample.KvCache)))
        {
            var runs = cell.Select(sample => sample.Run).Where(run => run.ExitCode == 0).ToArray();
            var first = runs.FirstOrDefault();
            var reference = samples.FirstOrDefault(sample => !sample.Warmup && sample.Context == cell.Key.Context &&
                $"{sample.Subject}/{sample.KvCache}" == referenceCell &&
                sample.Run.ExitCode == 0);
            rows.Add(new SweepRow(
                cell.Key.Context,
                cell.Key.Subject,
                cell.Key.KvCache,
                first?.PromptTokens,
                first?.TokenIdsIdentical,
                model.WeightsBytes / Mebibyte,
                model.KvBytesPerToken(cell.Key.KvCache) * cell.Key.Context / Mebibyte,
                Median(runs.Select(run => run.PeakFootprintBytes / Mebibyte)),
                Median(runs.Select(run => run.PeakResidentBytes / Mebibyte)),
                Median(runs.Select(run => run.TimeToFirstTokenMilliseconds)),
                Median(runs.Select(run => run.GenerationMilliseconds)),
                Median(runs.Select(run => run.DecodeTokensPerSecond)),
                first?.GeneratedTokens ?? 0,
                cell.Count(sample => sample.Correct && sample.Run.ExitCode == 0),
                cell.Count(),
                cell.Count() - runs.Length,
                reference is null || first is null ? null : MatchingPrefix(tokenizer, reference.Run.Output, first.Output)));
        }

        return rows;
    }

    public static string Markdown(IReadOnlyList<SweepRow> rows)
    {
        var text = new StringBuilder();
        _ = text.AppendLine("| context | engine | KV | KV MiB | peak RSS MiB | peak footprint MiB | TTFT s | full generation s | decode tok/s | tokens | correct | same tokens as reference |")
            .AppendLine("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        foreach (var row in rows.OrderBy(row => row.Context).ThenBy(row => row.Subject, StringComparer.Ordinal)
            .ThenBy(row => row.KvCache, StringComparer.Ordinal))
        {
            _ = text.AppendLine(string.Create(CultureInfo.InvariantCulture,
                $"| {row.Context} | {row.Subject} | {row.KvCache} | {Format(row.KvCacheMebibytes, "F0")} | " +
                $"{Format(row.PeakResidentMebibytes, "F0")} | " +
                $"{Format(row.PeakFootprintMebibytes, "F0")} | {Format(row.TimeToFirstTokenMilliseconds / 1000, "F2")} | " +
                $"{Format(row.GenerationMilliseconds / 1000, "F2")} | {Format(row.DecodeTokensPerSecond, "F1")} | " +
                $"{row.GeneratedTokens} | {row.CorrectRuns}/{row.MeasuredRuns}{(row.FailedRuns > 0 ? $", {row.FailedRuns} failed" : string.Empty)} | {Format(row.TokensMatchingReference, "F0")} |"));
        }

        return text.ToString();
    }

    /// <summary>Leading output tokens two engines share, both re-tokenized by the repo tokenizer.</summary>
    internal static int MatchingPrefix(ITextTokenizer tokenizer, string reference, string actual)
    {
        var left = tokenizer.Encode(reference, parseSpecialTokens: false);
        var right = tokenizer.Encode(actual, parseSpecialTokens: false);
        var count = 0;
        while (count < left.Count && count < right.Count && left[count] == right[count])
        {
            count++;
        }

        return count;
    }

    private static double? Median(IEnumerable<double?> values)
    {
        var present = values.Where(value => value is not null).Select(value => value!.Value).Order().ToArray();
        return present.Length == 0
            ? null
            : present.Length % 2 == 1
                ? present[present.Length / 2]
                : (present[(present.Length / 2) - 1] + present[present.Length / 2]) / 2;
    }

    private static string Format(double? value, string format) =>
        value is { } present ? present.ToString(format, CultureInfo.InvariantCulture) : "—";
}
