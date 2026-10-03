using System.Globalization;
using System.Text.Json;

internal sealed record PerformanceModelDescriptor(string Family, string Label,
    string WeightsSha256, string ScenarioSha256, string Execution, string Hardware)
{
    // REQ-WEB-003: identity comes from evidence, never from a workflow default.
    public static PerformanceModelDescriptor? Read(JsonElement root, string kind, string subject)
    {
        var cpu = kind is "smoke" or "cpu";
        var sample = cpu ? root.GetProperty("samples").EnumerateArray()
            .First(item => Text(item, "subject") == subject) : root.GetProperty("samples").EnumerateArray().First();
        var source = cpu ? sample.GetProperty("subject_result") : root;
        var path = Text(source, "model_path");
        var prepared = subject == "synapse" && path?.EndsWith(".synapse", StringComparison.OrdinalIgnoreCase) == true;
        if (cpu)
        {
            path = root.GetProperty("samples").EnumerateArray()
                .Select(item => Text(item.GetProperty("subject_result"), "model_path"))
                .FirstOrDefault(value => value?.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) == true)
                ?? path;
        }

        var label = kind == "foundry" ? Text(root, "variant_id") : path?.Replace('\\', '/').Split('/')[^1];
        if (string.IsNullOrWhiteSpace(label) || label.Length > 2048)
        {
            return null;
        }

        var family = kind == "foundry" ? Text(root, "alias") : FindFamily(label);
        var hash = Text(root, cpu ? "model_sha256" : "model_weights_sha256") ?? "not_reported";
        var scenario = Text(root, "scenario_sha256") ?? "not_reported";
        var execution = cpu
            ? $"{(subject == "synapse" ? (Text(root, "synapse_backend") ?? "CPU") + (prepared ? " · prepared .synapse" : "") : "CPU")} · " +
              $"{root.GetProperty("threads").GetInt32().ToString(CultureInfo.InvariantCulture)} threads · fresh process"
            : kind == "mlx" ? "Metal · MLX weights · resident model · " + (Text(root, "context_mode") ?? "cache policy not reported")
            : "CPU · ONNX weights · runtime-default threads · " + (Text(root, "context_mode") ?? "cache policy not reported");
        return new(family ?? label, label, hash, scenario, execution,
            Text(root, "runner_label") ?? "Runner details not reported");
    }

    private static string FindFamily(string label)
    {
        var lower = label.ToLowerInvariant();
        foreach (var family in new[] { "qwen2.5-0.5b", "qwen2.5-7b", "qwen3-0.6b",
            "phi-3.5-mini", "phi-4-mini", "mistral-7b-v0.2", "deepseek-r1-7b" })
        {
            if (lower.StartsWith(family + "-", StringComparison.Ordinal) || lower == family)
            {
                return family;
            }
        }

        return label;
    }

    private static string? Text(JsonElement root, string name) =>
        root.TryGetProperty(name, out var node) && node.ValueKind == JsonValueKind.String
            ? node.GetString() : null;
}
