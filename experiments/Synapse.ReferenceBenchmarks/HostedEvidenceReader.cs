using System.Globalization;
using System.Text.Json;

internal static class HostedEvidenceReader
{
    private static readonly string[] CpuSubjects = ["synapse", "dotllm", "llamasharp", "llamacpp"];

    public static IReadOnlyList<HostedResultRow> Read(JsonElement root, HostedArtifactSpec artifact, string file)
    {
        var smoke = artifact.Kind == "smoke";
        var dialogue = file.Contains("dialogue", StringComparison.Ordinal);
        var turns = dialogue ? 3 : 1;
        var maxTokens = smoke ? 8 : dialogue ? 64 : 128;
        if (root.GetProperty("max_tokens").GetInt32() != maxTokens ||
            (!smoke && root.GetProperty("turns").GetArrayLength() != turns))
        {
            throw new InvalidDataException("scenario or output-token limit mismatch");
        }

        if (artifact.Kind == "foundry" &&
            (root.GetProperty("alias").GetString() != artifact.Alias ||
             root.GetProperty("model_set_id").GetString() != artifact.ModelSetId ||
             root.GetProperty("device").GetString() != "cpu"))
        {
            throw new InvalidDataException("alias mismatch, model-set mismatch, or non-CPU variant");
        }

        if (artifact.Kind != "mlx")
        {
            var os = root.GetProperty("operating_system").GetString() ?? string.Empty;
            var validOs = artifact.Name.Contains("osx", StringComparison.Ordinal) ||
                          artifact.Name.Contains("macos", StringComparison.Ordinal)
                ? os.Contains("macOS", StringComparison.OrdinalIgnoreCase)
                : artifact.Name.Contains("linux", StringComparison.Ordinal) ||
                  artifact.Name.Contains("ubuntu", StringComparison.Ordinal)
                    ? os.Contains("Ubuntu", StringComparison.OrdinalIgnoreCase)
                    : os.Contains("Windows", StringComparison.OrdinalIgnoreCase);
            if (!validOs)
            {
                throw new InvalidDataException("runner operating-system mismatch");
            }
        }

        var measurements = root.GetProperty("measurements").GetInt32();
        var samples = root.GetProperty("samples").EnumerateArray()
            .Where(sample => !sample.GetProperty("warmup").GetBoolean()).ToArray();
        var subjectCount = artifact.Kind is "smoke" or "cpu" ? CpuSubjects.Length : 1;
        if (measurements <= 0 || samples.Length != measurements * turns * subjectCount)
        {
            throw new InvalidDataException("measured sample count mismatch");
        }

        var groups = samples.GroupBy(sample => (
            Turn: smoke ? 1 : sample.GetProperty("turn").GetInt32(),
            Subject: artifact.Kind is "smoke" or "cpu"
                ? sample.GetProperty("subject").GetString() ?? string.Empty
                : artifact.Kind == "foundry" ? artifact.Alias! : "SwiftLM Qwen2.5 0.5B"));
        var grouped = groups.ToArray();
        if (grouped.Length != turns * subjectCount || grouped.Any(group =>
                group.Count() != measurements || group.Key.Turn is < 1 or > 3 ||
                (subjectCount > 1 && !CpuSubjects.Contains(group.Key.Subject, StringComparer.Ordinal))))
        {
            throw new InvalidDataException("measured subject, turn, or round mismatch");
        }

        return [.. grouped.Select(group => Row(root, artifact, dialogue, group.Key.Turn,
            group.Key.Subject, [.. group]))];
    }

    private static HostedResultRow Row(JsonElement root, HostedArtifactSpec artifact,
        bool dialogue, int turn, string subject, JsonElement[] samples)
    {
        var cpu = artifact.Kind is "smoke" or "cpu";
        var smoke = artifact.Kind == "smoke";
        var runner = artifact.Kind == "foundry"
            ? Text(root, "runner_label") ?? "unknown"
            : artifact.Name.EndsWith("osx-arm64", StringComparison.Ordinal)
                ? "macOS 15 ARM64"
                : artifact.Name.EndsWith("linux-x64", StringComparison.Ordinal)
                    ? "Ubuntu 24.04 x64" : "Windows Server 2025 x64";
        var scenario = smoke ? "8-token smoke" : dialogue ? "3-turn dialogue" : "128-token answer";
        var state = smoke
            ? samples.All(sample => sample.GetProperty("quality_matched").GetBoolean())
                ? "matched" : "mismatch"
            : samples.All(sample => (Text(sample, "text") is null or "") &&
                !string.IsNullOrWhiteSpace(Text(sample, "reasoning_text")))
                ? "unreviewed; reasoning only" : "unreviewed";
        var finishes = samples.Select(sample => Text(sample, "finish_reason"))
            .Where(value => value is not null).Distinct(StringComparer.Ordinal).ToArray();
        if (finishes.Length > 0)
        {
            state += "; " + string.Join(',', finishes);
        }

        return new HostedResultRow(
            cpu ? 0 : artifact.Kind == "mlx" ? 1 : 2,
            cpu ? "GGUF CPU, fresh process" : artifact.Kind == "mlx"
                ? "MLX Metal, resident" : "Foundry Local, resident",
            runner, smoke ? 0 : dialogue ? 2 : 1, scenario, turn, subject, artifact.Name,
            samples.Length,
            Median(samples, sample => OutputTokens(sample, smoke), "0.#"),
            Median(samples, sample => Number(cpu ? sample.GetProperty("subject_result") : sample,
                "time_to_first_token_milliseconds")),
            Median(samples, sample => Number(cpu ? sample.GetProperty("subject_result") : sample,
                cpu && subject == "llamacpp" ? "native_eval_tokens_per_second" :
                "decode_tokens_per_second")),
            Median(samples, sample => Number(sample, smoke ? "matrix_process_wall_milliseconds"
                : cpu ? "process_wall_milliseconds" : "request_wall_milliseconds")),
            Median(samples, sample => Number(sample, artifact.Kind == "foundry"
                ? "resident_bytes" : "peak_resident_bytes") is { } bytes
                ? bytes / 1048576d : null),
            state);
    }

    private static double? OutputTokens(JsonElement sample, bool smoke)
    {
        if (!smoke)
        {
            return Number(sample, "generated_tokens");
        }

        var generated = sample.GetProperty("subject_result").GetProperty("generated_tokens");
        return generated.ValueKind == JsonValueKind.Array ? generated.GetArrayLength()
            : generated.ValueKind == JsonValueKind.Number ? generated.GetDouble() : null;
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() : null;

    private static double? Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble() : null;

    private static string Median(IEnumerable<JsonElement> samples, Func<JsonElement, double?> select,
        string format = "0.0")
    {
        var sorted = samples.Select(select).Where(value => value.HasValue)
            .Select(value => value!.Value).Order().ToArray();
        if (sorted.Length == 0)
        {
            return "n/a";
        }

        var middle = sorted.Length / 2;
        var value = sorted.Length % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
        return value.ToString(format, CultureInfo.InvariantCulture);
    }
}
