using System.Globalization;
using System.Text;
using System.Text.Json;

internal static class DialogueReportCommand
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length is not (2 or 4) || args[0] != "--input" ||
            (args.Length == 4 && args[2] != "--summary"))
        {
            Console.Error.WriteLine("Usage: report-dialogue --input <raw.json> [--summary <path>]");
            return 2;
        }

        try
        {
            await using var stream = File.OpenRead(args[1]);
            using var document = await JsonDocument.ParseAsync(stream).ConfigureAwait(false);
            var report = Render(document.RootElement);
            Console.Write(report);
            if (args.Length == 4)
            {
                await File.AppendAllTextAsync(args[3], report).ConfigureAwait(false);
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
            throw new InvalidDataException("No measured dialogue samples.");
        }

        var mlx = !samples[0].TryGetProperty("subject", out _);
        var result = new StringBuilder();
        _ = result.AppendLine("## One-/three-turn performance diagnostic — no winner verdict")
            .AppendLine()
            .Append("Status: `").Append(root.GetProperty("status").GetString())
            .Append("`; context: `").Append(root.GetProperty("context_mode").GetString())
            .AppendLine("`. Answer quality is not yet reviewed.  ");
        _ = result.Append("Model SHA-256: `")
            .Append(root.GetProperty(mlx ? "model_weights_sha256" : "model_sha256").GetString())
            .Append("`; maximum output tokens/turn: ").Append(root.GetProperty("max_tokens").GetInt32())
            .Append("; warm-ups: ").Append(root.GetProperty("warmups").GetInt32())
            .Append("; measured rounds: ").Append(root.GetProperty("measurements").GetInt32())
            .AppendLine(".  ");
        _ = result.AppendLine("| Turn | Subject | Prompt tokens | Output tokens | Load ms | TTFT ms | Decode tok/s | Request/process wall ms | Output tok/s incl. wall | Process CPU ms | Avg CPU cores | Peak RSS MiB | Mac footprint MiB | Observed cache-hit tokens |")
            .AppendLine("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        foreach (var group in samples.GroupBy(sample => (
            Turn: sample.GetProperty("turn").GetInt32(),
            Subject: mlx ? "swiftlm-mlx-metal" : sample.GetProperty("subject").GetString() ?? string.Empty))
            .OrderBy(group => group.Key.Turn).ThenBy(group => group.Key.Subject))
        {
            var groupSamples = group.ToArray();
            _ = result.Append('|').Append(group.Key.Turn).Append('|').Append(group.Key.Subject).Append('|')
                .Append(Median(groupSamples.Select(sample => PromptTokens(root, sample, mlx))))
                .Append('|').Append(Median(groupSamples.Select(sample => Number(sample, "generated_tokens"))))
                .Append('|').Append(Median(groupSamples.Select(sample => Phase(sample, mlx, "load_milliseconds"))))
                .Append('|').Append(Median(groupSamples.Select(sample => Phase(sample, mlx, "time_to_first_token_milliseconds"))))
                .Append('|').Append(Median(groupSamples.Select(sample => Phase(sample, mlx, "decode_tokens_per_second"))))
                .Append('|').Append(Median(groupSamples.Select(sample => Wall(sample, mlx))))
                .Append('|').Append(Median(groupSamples.Select(sample => OutputRate(sample, mlx))))
                .Append('|').Append(Median(groupSamples.Select(sample => Number(sample, "process_cpu_milliseconds"))))
                .Append('|').Append(Median(groupSamples.Select(sample => AverageCpuCores(sample, mlx))))
                .Append('|').Append(Median(groupSamples.Select(sample => Bytes(sample, "peak_resident_bytes"))))
                .Append('|').Append(Median(groupSamples.Select(sample => Bytes(sample,
                    mlx ? "physical_footprint_bytes" : "peak_physical_footprint_bytes"))))
                .Append('|').Append(mlx ? Median(groupSamples.Select(sample => Number(sample, "cache_hit_tokens"))) : "n/a")
                .AppendLine("|");
        }

        if (mlx)
        {
            var cold = root.GetProperty("samples").EnumerateArray()
                .FirstOrDefault(sample => sample.GetProperty("warmup").GetBoolean() &&
                    sample.GetProperty("turn").GetInt32() == 1);
            if (cold.ValueKind == JsonValueKind.Object)
            {
                _ = result.AppendLine()
                    .Append("First fresh-server request (warm-up, excluded from medians): TTFT ")
                    .Append(Median([Number(cold, "time_to_first_token_milliseconds")]))
                    .Append(" ms; observed cache hit ")
                    .Append(Median([Number(cold, "cache_hit_tokens")]))
                    .AppendLine(" tokens. This is not an isolated cache speedup estimate.");
            }
        }

        _ = result.AppendLine().AppendLine(mlx
            ? "MLX uses a resident Metal server and separately quantized weights. Cache-hit tokens come from the pinned server's log; `n/a` means no hit was observed, not proof of a miss. Request wall excludes one-time server load. RSS and physical footprint are separate OS views and must not be added; GPU allocation is not separately measured. Do not rank this cohort against CPU GGUF."
            : "Each CPU turn starts a fresh process with the locked earlier answers in its prompt. No KV or prefix-cache reuse is measured. Process wall includes load; native llama.cpp does not expose comparable load/TTFT/decode phases. CPU and RSS cover the direct subject process, not a CLR-only heap.");
        _ = result.AppendLine("Medians are per column; raw per-request JSON retains the original text, token counts, and scopes.");
        return result.ToString();
    }

    private static double? PromptTokens(JsonElement root, JsonElement sample, bool mlx)
    {
        if (mlx)
        {
            return Number(sample, "prompt_tokens");
        }

        var index = sample.GetProperty("turn").GetInt32() - 1;
        return root.GetProperty("turns")[index].GetProperty("prompt_token_ids").GetArrayLength();
    }

    private static double? Phase(JsonElement sample, bool mlx, string name) =>
        Number(mlx ? sample : sample.GetProperty("subject_result"), name);

    private static double? Wall(JsonElement sample, bool mlx) =>
        Number(sample, mlx ? "request_wall_milliseconds" : "process_wall_milliseconds");

    private static double? OutputRate(JsonElement sample, bool mlx)
    {
        var tokens = Number(sample, "generated_tokens");
        var wall = Wall(sample, mlx);
        return tokens is { } count && wall is > 0 ? count * 1000 / wall : null;
    }

    private static double? AverageCpuCores(JsonElement sample, bool mlx)
    {
        var cpu = Number(sample, "process_cpu_milliseconds");
        var wall = Wall(sample, mlx);
        return cpu is { } milliseconds && wall is > 0 ? milliseconds / wall : null;
    }

    private static double? Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble() : null;

    private static double? Bytes(JsonElement element, string name) =>
        Number(element, name) is { } value ? value / 1048576d : null;

    private static string Median(IEnumerable<double?> values)
    {
        var sorted = values.Where(value => value.HasValue).Select(value => value!.Value)
            .OrderBy(value => value).ToArray();
        if (sorted.Length == 0)
        {
            return "n/a";
        }

        var middle = sorted.Length / 2;
        var value = sorted.Length % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
        return value.ToString("0.0", Invariant);
    }
}
