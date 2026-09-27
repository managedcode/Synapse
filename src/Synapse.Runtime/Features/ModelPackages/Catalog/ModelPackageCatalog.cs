using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;

/// <summary>A validated catalog of reproducible model packages.</summary>
public sealed class ModelPackageCatalog
{
    private const int SupportedSchemaVersion = 1;
    private const int MaximumCatalogBytes = 1024 * 1024;
    private readonly ReadOnlyDictionary<string, ModelPackageDefinition> packagesById;

    private ModelPackageCatalog(IEnumerable<ModelPackageDefinition> packages)
    {
        Packages = Array.AsReadOnly([.. packages]);
        packagesById = new ReadOnlyDictionary<string, ModelPackageDefinition>(
            Packages.ToDictionary(package => package.Id, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>Packages in deterministic catalog order.</summary>
    public IReadOnlyList<ModelPackageDefinition> Packages { get; }

    /// <summary>Loads and validates a source-controlled catalog.</summary>
    public static ModelPackageCatalog Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var file = new FileInfo(Path.GetFullPath(path));
        if (!file.Exists || file.Length is <= 0 or > MaximumCatalogBytes)
        {
            throw new InvalidDataException(
                $"Model catalog must exist and be inside [1, {MaximumCatalogBytes}] bytes.");
        }

        var json = File.ReadAllText(file.FullName);
        var document = JsonSerializer.Deserialize(json, ModelCatalogJsonContext.Default.CatalogDocument)
            ?? throw new InvalidDataException("Model catalog is empty.");
        if (document.SchemaVersion != SupportedSchemaVersion)
        {
            throw new InvalidDataException(
                $"Model catalog schema {document.SchemaVersion} is unsupported.");
        }

        var packages = document.Packages.Select(ModelPackageDefinition.Create).ToArray();
        if (packages.Length == 0 ||
            packages.Select(package => package.Id).Distinct(StringComparer.OrdinalIgnoreCase).Count() != packages.Length)
        {
            throw new InvalidDataException("Model catalog must contain packages with unique IDs.");
        }

        return new ModelPackageCatalog(packages);
    }

    /// <summary>Selects exactly one package by ID.</summary>
    public ModelPackageDefinition GetRequired(string id) => packagesById.TryGetValue(id, out var package)
        ? package
        : throw new KeyNotFoundException($"Model package '{id}' is not present in the catalog.");

    /// <summary>Selects every package assigned to a named set.</summary>
    public IReadOnlyList<ModelPackageDefinition> SelectSet(string setName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(setName);
        var selected = Packages.Where(package => package.Sets.Contains(setName, StringComparer.Ordinal)).ToArray();
        return selected.Length > 0
            ? Array.AsReadOnly(selected)
            : throw new KeyNotFoundException($"Model set '{setName}' is not present in the catalog.");
    }
}

/// <summary>One immutable model source and its verified files.</summary>
public sealed class ModelPackageDefinition
{
    private ModelPackageDefinition(CatalogPackage source, IReadOnlyList<ModelPackageFile> files)
    {
        Id = source.Id;
        DisplayName = source.DisplayName;
        Architecture = source.Architecture;
        Kind = source.Kind;
        Format = source.Format;
        Precision = source.Precision;
        ParameterCount = source.ParameterCount;
        License = source.License;
        SourceUrl = source.SourceUrl;
        Revision = source.Revision;
        Sets = Array.AsReadOnly([.. source.Sets]);
        Files = files;
    }

    /// <summary>Stable package selector and local directory name.</summary>
    public string Id { get; }
    /// <summary>Human-readable upstream model name.</summary>
    public string DisplayName { get; }
    /// <summary>Executor architecture key, independent of branding.</summary>
    public string Architecture { get; }
    /// <summary>Workload kind such as text generation or embeddings.</summary>
    public string Kind { get; }
    /// <summary>Source container format.</summary>
    public string Format { get; }
    /// <summary>Declared source storage precision.</summary>
    public string Precision { get; }
    /// <summary>Declared total parameter count.</summary>
    public long ParameterCount { get; }
    /// <summary>Upstream model license identifier.</summary>
    public string License { get; }
    /// <summary>Canonical upstream model page.</summary>
    public string SourceUrl { get; }
    /// <summary>Immutable upstream source revision.</summary>
    public string Revision { get; }
    /// <summary>Named download and verification sets.</summary>
    public IReadOnlyList<string> Sets { get; }
    /// <summary>All content-addressed package files.</summary>
    public IReadOnlyList<ModelPackageFile> Files { get; }

    internal static ModelPackageDefinition Create(CatalogPackage source)
    {
        ValidateIdentifier(source.Id, "package ID");
        RequireText(source.DisplayName, "display name");
        ValidateIdentifier(source.Architecture, "architecture");
        ValidateIdentifier(source.Kind, "kind");
        ValidateIdentifier(source.Format, "format");
        RequireText(source.Precision, "precision");
        RequireText(source.License, "license");
        RequireHttps(source.SourceUrl, "source URL");
        if (source.ParameterCount <= 0 || source.Revision.Length != 40 ||
            !source.Revision.All(Uri.IsHexDigit) || source.Sets.Count == 0 ||
            source.Sets.Any(string.IsNullOrWhiteSpace) || source.Sets.Distinct().Count() != source.Sets.Count)
        {
            throw new InvalidDataException($"Model package '{source.Id}' has invalid metadata.");
        }

        var files = source.Files.Select(file => ModelPackageFile.Create(source.Id, file)).ToArray();
        if (files.Length == 0 ||
            files.Select(file => file.Path).Distinct(StringComparer.OrdinalIgnoreCase).Count() != files.Length)
        {
            throw new InvalidDataException($"Model package '{source.Id}' must contain unique files.");
        }

        return new ModelPackageDefinition(source, Array.AsReadOnly(files));
    }

    private static void ValidateIdentifier(string value, string field)
    {
        RequireText(value, field);
        if (value[0] == '.' ||
            value.Any(character => !(char.IsAsciiLetterOrDigit(character) || character is '-' or '_' or '.')))
        {
            throw new InvalidDataException($"Model {field} '{value}' contains unsafe characters.");
        }
    }

    private static void RequireText(string value, string field)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidDataException($"Model {field} cannot be empty.");
        }
    }

    private static void RequireHttps(string value, string field)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps)
        {
            throw new InvalidDataException($"Model {field} must use HTTPS.");
        }
    }
}

/// <summary>One content-addressed file in a model package.</summary>
public sealed record ModelPackageFile(string Path, Uri Url, long SizeBytes, string Sha256)
{
    internal static ModelPackageFile Create(string packageId, CatalogFile source)
    {
        if (string.IsNullOrWhiteSpace(source.Path) || System.IO.Path.IsPathRooted(source.Path) ||
            source.Path.Split('/', StringSplitOptions.RemoveEmptyEntries).Any(segment => segment is "." or "..") ||
            source.Path.Contains('\\', StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Model package '{packageId}' contains an unsafe file path.");
        }

        if (!Uri.TryCreate(source.Url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps ||
            source.SizeBytes <= 0 || source.Sha256.Length != 64 || !source.Sha256.All(Uri.IsHexDigit))
        {
            throw new InvalidDataException($"Model package '{packageId}' contains invalid file metadata.");
        }

        return new ModelPackageFile(source.Path, uri, source.SizeBytes, source.Sha256.ToLowerInvariant());
    }
}

internal sealed class CatalogDocument
{
    public int SchemaVersion { get; init; }
    public List<CatalogPackage> Packages { get; init; } = [];
}

internal sealed class CatalogPackage
{
    public string Id { get; init; } = string.Empty;
    public string DisplayName { get; init; } = string.Empty;
    public string Architecture { get; init; } = string.Empty;
    public string Kind { get; init; } = string.Empty;
    public string Format { get; init; } = string.Empty;
    public string Precision { get; init; } = string.Empty;
    public long ParameterCount { get; init; }
    public string License { get; init; } = string.Empty;
    public string SourceUrl { get; init; } = string.Empty;
    public string Revision { get; init; } = string.Empty;
    public List<string> Sets { get; init; } = [];
    public List<CatalogFile> Files { get; init; } = [];
}

internal sealed class CatalogFile
{
    public string Path { get; init; } = string.Empty;
    public string Url { get; init; } = string.Empty;
    public long SizeBytes { get; init; }
    public string Sha256 { get; init; } = string.Empty;
}

[JsonSerializable(typeof(CatalogDocument))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class ModelCatalogJsonContext : JsonSerializerContext;
