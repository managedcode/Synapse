using System.Text.Json;
using System.Text.Json.Nodes;

internal static class PerformanceModelEnrichment
{
    public static async Task EnrichAsync(string root, JsonObject report, JsonArray issues)
    {
        var cache = new Dictionary<string, JsonObject[]>(StringComparer.Ordinal);
        foreach (var row in report["rows"]!.AsArray().OfType<JsonObject>())
        {
            if (row["model"] is not null)
            {
                continue;
            }

            var name = row["source_artifact"]!.GetValue<string>();
            if ((!name.StartsWith("performance-", StringComparison.Ordinal) &&
                 !name.StartsWith("foundry-local-", StringComparison.Ordinal)) || name.Length > 150 ||
                name.Any(c => !char.IsAsciiLetterOrDigit(c) && c is not '-' and not '.'))
            {
                issues.Add("Model identity unavailable: invalid artifact name.");
                continue;
            }

            var kind = name.StartsWith("foundry-local-", StringComparison.Ordinal) ? "foundry"
                : name == "performance-mlx-osx-arm64" ? "mlx"
                : name.StartsWith("performance-long-", StringComparison.Ordinal) ? "cpu" : "smoke";
            var dialogue = row["scenario"]!.GetValue<string>() == "3-turn dialogue";
            var file = kind == "smoke" ? "synapse-benchmark.json"
                : $"{(kind == "foundry" ? "foundry" : kind == "mlx" ? "mlx" : "synapse")}-{(dialogue ? "dialogue" : "single")}.json";
            var path = Path.Combine(root, "performance", name, file);
            try
            {
                if (!cache.TryGetValue(path, out var projections))
                {
                    projections = await ReadAsync(path, name, kind, file).ConfigureAwait(false);
                    cache.Add(path, projections);
                }

                var projection = projections.SingleOrDefault(candidate => row.All(property =>
                    property.Key == "model" || JsonNode.DeepEquals(property.Value, candidate[property.Key])));
                if (projection?["model"] is { } model)
                {
                    row["model"] = model.DeepClone();
                }
                else
                {
                    issues.Add($"{name}/{file}: model identity unavailable; raw metrics do not match summary.");
                }
            }
            catch (Exception exception) when (exception is IOException or InvalidDataException or
                JsonException or InvalidOperationException or KeyNotFoundException)
            {
                issues.Add($"{name}/{file}: model identity unavailable ({exception.Message}).");
            }
        }
    }

    private static async Task<JsonObject[]> ReadAsync(string path, string name, string kind, string file)
    {
        var raw = await SitePublicationValidation.ReadAsync(path).ConfigureAwait(false);
        using var document = JsonDocument.Parse(raw.ToJsonString());
        var source = document.RootElement;
        var alias = kind == "foundry" ? source.GetProperty("alias").GetString() : null;
        if (alias is not null && !name.EndsWith('-' + alias, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Model alias does not match the artifact.");
        }

        var spec = new HostedArtifactSpec(name, kind, [file], alias,
            kind == "foundry" ? source.GetProperty("model_set_id").GetString() : null);
        return [.. HostedEvidenceReader.Read(source, spec, file).Select(row =>
            JsonSerializer.SerializeToNode(PerformancePublicationJson.Project(row),
                PerformancePublicationJsonContext.Default.PerformancePublicationRow)!.AsObject())];
    }
}
