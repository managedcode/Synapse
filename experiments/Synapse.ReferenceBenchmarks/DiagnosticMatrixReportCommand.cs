using System.Globalization;
using System.Text;
using System.Text.Json;

internal static class DiagnosticMatrixReportCommand
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;
    private static readonly string[] SubjectOrder = ["synapse", "dotllm", "llamasharp", "llamacpp"];

    public static async Task<int> RunAsync(string[] arguments)
    {
        if (arguments.Length is not (2 or 4) || arguments[0] != "--input" ||
            (arguments.Length == 4 && arguments[2] != "--summary"))
        {
            Console.Error.WriteLine("Usage: report --input <matrix.json> [--summary <github-step-summary>]");
            return 2;
        }

        try
        {
            await using var stream = File.OpenRead(arguments[1]);
            using var document = await JsonDocument.ParseAsync(stream).ConfigureAwait(false);
            var report = Render(document.RootElement);
            Console.Write(report);
            if (arguments.Length == 4)
            {
                await File.AppendAllTextAsync(arguments[3], report).ConfigureAwait(false);
            }

            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static string Render(JsonElement root)
    {
        var samples = root.GetProperty("samples").EnumerateArray()
            .Where(sample => !sample.GetProperty("warmup").GetBoolean()).ToArray();
        if (samples.Length == 0)
        {
            throw new InvalidDataException("Matrix evidence has no measured samples.");
        }

        var result = new StringBuilder();
        _ = result.AppendLine("## Performance diagnostic — no winner verdict").AppendLine();
        _ = result.Append("Status: `").Append(root.GetProperty("status").GetString())
            .AppendLine("`. Runner timing is descriptive, not a cross-machine ranking.").AppendLine();
        var modelPath = samples[0].GetProperty("subject_result").GetProperty("model_path").GetString();
        var modelFile = modelPath?.Replace('\\', '/').Split('/').LastOrDefault();
        _ = result.Append("Model: `").Append(Escape(modelFile))
            .Append("`; SHA-256: `").Append(root.GetProperty("model_sha256").GetString())
            .AppendLine("`.  ");
        _ = result.Append("OS/architecture: ").Append(root.GetProperty("operating_system").GetString())
            .Append(" / ").Append(root.GetProperty("architecture").GetString()).AppendLine(".  ");
        if (root.TryGetProperty("runner_label", out var runner))
        {
            _ = result.Append("Runner label: `").Append(Escape(runner.GetString()))
                .Append("`; visible logical processors: ")
                .Append(root.GetProperty("logical_processors").GetInt32()).AppendLine(".  ");
        }
        _ = result.Append("Prompt: `").Append(Escape(root.GetProperty("prompt").GetString()))
            .Append("`; prompt IDs: `").Append(Ids(root.GetProperty("prompt_token_ids")))
            .Append("`; expected IDs: `").Append(Ids(root.GetProperty("expected_token_ids")))
            .AppendLine("`.  ");
        _ = result.Append("Expected text: `").Append(Escape(root.GetProperty("expected_text").GetString()))
            .Append("`; output tokens: ").Append(root.GetProperty("max_tokens").GetInt32())
            .Append("; thread cap: ").Append(root.GetProperty("threads").GetInt32())
            .Append("; warm-ups: ").Append(root.GetProperty("warmups").GetInt32())
            .Append("; Measured rounds: ").Append(root.GetProperty("measurements").GetInt32())
            .AppendLine(".  ");
        _ = result.Append("Power state: `").Append(root.GetProperty("power_state").GetString())
            .AppendLine("`. Each sample is a fresh process; order rotates by round.").AppendLine();
        _ = result.AppendLine("| Subject | Quality | Median process wall ms | Median peak RSS MiB | Median physical footprint MiB | Median reported decode tok/s |")
            .AppendLine("|---|---|---:|---:|---:|---:|");

        foreach (var group in samples.GroupBy(sample => sample.GetProperty("subject").GetString())
            .OrderBy(group => Array.IndexOf(SubjectOrder, group.Key)))
        {
            var subjectSamples = group.ToArray();
            var quality = subjectSamples.All(sample => sample.GetProperty("quality_matched").GetBoolean())
                ? "matched" : "mismatch";
            _ = result.Append('|').Append(group.Key).Append('|').Append(quality).Append('|')
                .Append(Median(subjectSamples.Select(sample => Number(sample, "matrix_process_wall_milliseconds"))))
                .Append('|').Append(Median(subjectSamples.Select(sample => Mebibytes(sample, "peak_resident_bytes"))))
                .Append('|').Append(Median(subjectSamples.Select(sample => Mebibytes(sample, "peak_physical_footprint_bytes"))))
                .Append('|').Append(Median(subjectSamples.Select(sample => Number(
                    sample.GetProperty("subject_result"), "decode_tokens_per_second"))))
                .AppendLine("|");
        }

        _ = result.AppendLine().AppendLine("RSS is whole-process resident memory, not CLR heap. Physical footprint is only available on macOS. Decode rates are subject-reported phases and are **not** comparable to whole-process wall time; native llama.cpp has no equivalent decode rate in this interface. Raw per-round JSON is the source of truth.");
        return result.ToString();
    }

    private static string Ids(JsonElement array) => string.Join(',', array.EnumerateArray()
        .Select(value => value.GetInt32().ToString(Invariant)));

    private static string Escape(string? value) => (value ?? string.Empty)
        .Replace("|", "\\|", StringComparison.Ordinal)
        .Replace("\n", "\\n", StringComparison.Ordinal);

    private static double? Number(JsonElement element, string property) =>
        element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble() : null;

    private static double? Mebibytes(JsonElement element, string property) =>
        Number(element, property) is { } bytes ? bytes / 1048576d : null;

    private static string Median(IEnumerable<double?> values)
    {
        var sorted = values.Where(value => value.HasValue).Select(value => value!.Value)
            .OrderBy(value => value).ToArray();
        if (sorted.Length == 0)
        {
            return "n/a";
        }

        var middle = sorted.Length / 2;
        var median = sorted.Length % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) / 2d : sorted[middle];
        return median.ToString("0.0", Invariant);
    }
}
