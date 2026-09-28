using System.Text.Json;
using System.Text.Json.Serialization;

/// <summary>A pinned Foundry Local model set and the hosted runners it targets (ADR-011).</summary>
internal sealed class FoundryModelSet
{
    private FoundryModelSet(string id, int contextTokens, IReadOnlyList<FoundryRunner> runners,
        IReadOnlyList<FoundryModel> models)
    {
        Id = id;
        ContextTokens = contextTokens;
        Runners = runners;
        Models = models;
    }

    public string Id { get; }

    public int ContextTokens { get; }

    public IReadOnlyList<FoundryRunner> Runners { get; }

    public IReadOnlyList<FoundryModel> Models { get; }

    public static FoundryModelSet Load(string path)
    {
        var document = JsonSerializer.Deserialize(File.ReadAllText(path),
            FoundryModelSetJsonContext.Default.FoundryModelSetDocument)
            ?? throw new InvalidDataException("The Foundry model set is empty.");
        if (document.SchemaVersion != 1 || string.IsNullOrWhiteSpace(document.Id) ||
            document.ContextTokens is < 256 or > 32768 ||
            document.Runners is not { Count: > 0 } || document.Models is not { Count: > 0 })
        {
            throw new InvalidDataException(
                "The Foundry model set needs schemaVersion 1, an id, contextTokens in [256, 32768], " +
                "runners, and models.");
        }

        ValidateRunners(document.Runners);
        var aliases = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var model in document.Models)
        {
            ValidateModel(model, document.Runners);
            if (!aliases.Add(model.Alias))
            {
                throw new InvalidDataException($"The Foundry model set has a duplicate alias '{model.Alias}'.");
            }
        }

        return new FoundryModelSet(document.Id, document.ContextTokens, document.Runners, document.Models);
    }

    public static bool Fits(FoundryVariant variant, FoundryRunner runner) =>
        (long)variant.FileSizeMb * 2 <= runner.MemoryMb;

    public FoundryModel GetModel(string alias) =>
        Models.FirstOrDefault(model => string.Equals(model.Alias, alias, StringComparison.OrdinalIgnoreCase))
        ?? throw new FoundryUsageException($"Model alias '{alias}' is not in set '{Id}'.");

    private static void ValidateRunners(List<FoundryRunner> runners)
    {
        if (runners.Any(runner => string.IsNullOrWhiteSpace(runner.Id) ||
                string.IsNullOrWhiteSpace(runner.Name) || runner.MemoryMb <= 0) ||
            runners.Select(runner => runner.Id).Distinct(StringComparer.Ordinal).Count() != runners.Count)
        {
            throw new InvalidDataException("Every runner needs a unique id, a name, and positive memoryMb.");
        }
    }

    private static void ValidateModel(FoundryModel model, List<FoundryRunner> runners)
    {
        if (string.IsNullOrWhiteSpace(model.Alias) || string.IsNullOrWhiteSpace(model.Family) ||
            model.Variants is not { Count: > 0 } ||
            model.Variants.Any(variant => variant.Device is not ("cpu" or "gpu") ||
                string.IsNullOrWhiteSpace(variant.Id) || variant.FileSizeMb <= 0) ||
            model.Variants.Select(variant => variant.Device).Distinct(StringComparer.Ordinal).Count() !=
                model.Variants.Count)
        {
            throw new InvalidDataException(
                $"Model '{model.Alias}' needs a family and unique cpu/gpu variants with ids and sizes.");
        }

        if (model.SystemPromptMode is not (FoundryModel.NativeSystemRole or FoundryModel.PrependToFirstUser))
        {
            throw new InvalidDataException(
                $"Model '{model.Alias}' has an unknown systemPrompt '{model.SystemPrompt}'.");
        }

        var cpu = model.Variants.FirstOrDefault(variant => variant.Device == "cpu")
            ?? throw new InvalidDataException($"Model '{model.Alias}' has no cpu variant.");
        if (!runners.Any(runner => Fits(cpu, runner)))
        {
            throw new InvalidDataException(
                $"Model '{model.Alias}' ({cpu.FileSizeMb} MB) fits no runner under the half-memory rule.");
        }
    }
}

internal sealed record FoundryRunner(string Id, string Name, int MemoryMb);

internal sealed record FoundryVariant(string Device, string Id, int FileSizeMb);

internal sealed record FoundryModel(string Family, string Alias, string? Note, string? SystemPrompt,
    List<FoundryVariant> Variants)
{
    public const string NativeSystemRole = "native";
    public const string PrependToFirstUser = "prepend-to-first-user";

    /// <summary>How the scenario system prompt reaches a chat template that may lack a system role.</summary>
    public string SystemPromptMode => SystemPrompt ?? NativeSystemRole;

    public FoundryVariant Variant(string device) =>
        Variants.FirstOrDefault(variant => variant.Device == device)
        ?? throw new FoundryUsageException($"Model '{Alias}' has no {device} variant.");
}

internal sealed record FoundryModelSetDocument(int SchemaVersion, string Id, int ContextTokens,
    string? ContextRule, string? MemoryRule, List<FoundryRunner> Runners, List<FoundryModel> Models);

[JsonSerializable(typeof(FoundryModelSetDocument))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class FoundryModelSetJsonContext : JsonSerializerContext;
