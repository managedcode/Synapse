using System.Globalization;
using System.Text;
using System.Text.Json;

internal static class FoundryReportCommand
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public static async Task<int> RunAsync(FoundryArguments arguments)
    {
        arguments.AllowOnly("--input", "--summary");
        await using var stream = File.OpenRead(arguments.Require("--input"));
        using var document = await JsonDocument.ParseAsync(stream).ConfigureAwait(false);
        var report = Render(document.RootElement);
        Console.Write(report);
        if (arguments.Optional("--summary") is { } summary)
        {
            await File.AppendAllTextAsync(summary, report).ConfigureAwait(false);
        }

        return 0;
    }

    private static string Render(JsonElement root)
    {
        var samples = root.GetProperty("samples").EnumerateArray()
            .Where(sample => !sample.GetProperty("warmup").GetBoolean()).ToArray();
        if (samples.Length == 0)
        {
            throw new InvalidDataException("No measured Foundry Local samples.");
        }

        var result = AppendHeader(new StringBuilder(), root)
            .AppendLine("| Turn | Prompt tokens | Output tokens | TTFT ms | Decode tok/s | Request wall ms | " +
                "Output tok/s incl. wall | Process CPU ms | Avg CPU cores | Peak RSS MiB |")
            .AppendLine("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        foreach (var turn in samples.GroupBy(sample => sample.GetProperty("turn").GetInt32()).OrderBy(g => g.Key))
        {
            var group = turn.ToArray();
            _ = result.Append('|').Append(turn.Key)
                .Append('|').Append(Median(group, sample => Number(sample, "prompt_tokens")))
                .Append('|').Append(Median(group, sample => Number(sample, "generated_tokens")))
                .Append('|').Append(Median(group, sample => Number(sample, "time_to_first_token_milliseconds")))
                .Append('|').Append(Median(group, sample => Number(sample, "decode_tokens_per_second")))
                .Append('|').Append(Median(group, sample => Number(sample, "request_wall_milliseconds")))
                .Append('|').Append(Median(group, sample => Ratio(sample, "generated_tokens", 1000)))
                .Append('|').Append(Median(group, sample => Number(sample, "process_cpu_milliseconds")))
                .Append('|').Append(Median(group, sample => Ratio(sample, "process_cpu_milliseconds", 1)))
                .Append('|').Append(Median(group, sample => Mebibytes(sample, "resident_bytes")))
                .AppendLine("|");
        }

        return result.AppendLine()
            .AppendLine("Foundry Local keeps the model loaded in one process and sends every request through a " +
                "new chat session with the full locked transcript, so no KV or prefix-cache reuse is measured. " +
                "Its ONNX weights, quantization, and chat template differ from the GGUF Q8_0 and MLX 8-bit " +
                "packages, and its thread count is the runtime default. Do not rank this cohort against the " +
                "GGUF or MLX cohorts. RSS and Mac footprint are separate OS views and must not be added.")
            .AppendLine("Medians are per column; the raw JSON keeps every request's text, reasoning text, and " +
                "token counts.")
            .AppendLine()
            .ToString();
    }

    private static StringBuilder AppendHeader(StringBuilder result, JsonElement root) => result
        .Append("## Foundry Local · ").Append(Text(root, "alias"))
        .AppendLine(" — separate ONNX cohort, no winner verdict").AppendLine()
        .Append("Model `").Append(Text(root, "variant_id")).Append("` (").Append(Text(root, "family"))
        .Append(", ").Append(Text(root, "device")).Append(", `").Append(Text(root, "execution_provider"))
        .Append("`, ").Append(Text(root, "catalog_file_size_mb")).Append(" MB); SDK ")
        .Append(Text(root, "sdk_version")).Append("; runner `").Append(Text(root, "runner_label"))
        .Append("` (").Append(Text(root, "operating_system")).Append(", ")
        .Append(Text(root, "processor_count")).AppendLine(" logical CPUs).  ")
        .Append("Status `").Append(Text(root, "status")).Append("`; context `")
        .Append(Text(root, "context_mode")).Append("`; context tokens ").Append(Text(root, "context_tokens"))
        .Append(" (package ").Append(Text(root, "original_max_length")).Append("); system prompt `")
        .Append(Text(root, "system_prompt_mode")).Append("`; threads `").Append(Text(root, "thread_policy"))
        .AppendLine("`.  ")
        .Append("Load ").Append(Format(Number(root, "load_milliseconds"))).Append(" ms; whole-process peak RSS ")
        .Append(Format(Mebibytes(root, "peak_resident_bytes"))).Append(" MiB; Mac footprint ")
        .Append(Format(Mebibytes(root, "peak_physical_footprint_bytes"))).Append(" MiB; max output tokens ")
        .Append(Text(root, "max_tokens")).Append("; warm-ups ").Append(Text(root, "warmups"))
        .Append("; measured rounds ").Append(Text(root, "measurements")).AppendLine(".").AppendLine();

    private static double? Ratio(JsonElement sample, string numerator, double scale)
    {
        var value = Number(sample, numerator);
        var wall = Number(sample, "request_wall_milliseconds");
        return value is { } amount && wall is > 0 ? amount * scale / wall.Value : null;
    }

    private static string Text(JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value))
        {
            return "n/a";
        }

        return value.ValueKind == JsonValueKind.String ? value.GetString() ?? "n/a"
            : value.ValueKind == JsonValueKind.Number ? value.GetRawText() : "n/a";
    }

    private static double? Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble() : null;

    private static double? Mebibytes(JsonElement element, string name) =>
        Number(element, name) is { } value ? value / 1048576d : null;

    private static string Format(double? value) => value?.ToString("0.0", Invariant) ?? "n/a";

    private static string Median(IEnumerable<JsonElement> samples, Func<JsonElement, double?> select)
    {
        var sorted = samples.Select(select).Where(value => value.HasValue).Select(value => value!.Value)
            .Order().ToArray();
        if (sorted.Length == 0)
        {
            return "n/a";
        }

        var middle = sorted.Length / 2;
        return Format(sorted.Length % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]);
    }
}
