using ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class ModelPackageCatalogTests
{
    [Test]
    public async Task RepositoryCatalogPinsDistinctArchitecturesAndWorkloadSets()
    {
        var catalog = ModelPackageCatalog.Load(Path.Combine(FindRepositoryRoot(), "models", "catalog.json"));

        await Assert.That(catalog.SelectSet("smoke").Select(package => package.Id))
            .IsEquivalentTo(["qwen2.5-0.5b-instruct-q8_0"]);
        await Assert.That(catalog.SelectSet("family-small").Select(package => package.Architecture))
            .IsEquivalentTo(["qwen2", "llama"]);
        await Assert.That(catalog.SelectSet("medium").Select(package => package.Architecture))
            .IsEquivalentTo(["phi3", "qwen2", "mistral3"]);
        await Assert.That(catalog.SelectSet("architecture-small").Select(package => package.Architecture))
            .IsEquivalentTo(["qwen3", "mamba"]);
        await Assert.That(catalog.SelectSet("embedding-small").Select(package => package.Kind))
            .IsEquivalentTo(["embeddings", "embeddings"]);
        await Assert.That(catalog.SelectSet("embedding-small").Select(package => package.Architecture))
            .IsEquivalentTo(["bert", "bert"]);
        await Assert.That(catalog.GetRequired("smollm2-135m-instruct-bf16").Files.Count)
            .IsEqualTo(8);
        await Assert.That(catalog.Packages.SelectMany(package => package.Files)
            .All(file => file.Url.Scheme == Uri.UriSchemeHttps && file.Sha256.Length == 64)).IsTrue();
    }

    [Test]
    public async Task CatalogRejectsPackagePathTraversal()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-catalog-{Guid.NewGuid():N}.json");
        const string json = /*lang=json,strict*/ """
            {
              "schemaVersion": 1,
              "packages": [{
                "id": "unsafe", "displayName": "Unsafe", "architecture": "qwen2",
                "kind": "text-generation", "format": "gguf", "precision": "Q8_0",
                "parameterCount": 1, "license": "MIT",
                "sourceUrl": "https://example.test/model",
                "revision": "1111111111111111111111111111111111111111",
                "sets": ["smoke"],
                "files": [{
                  "path": "../outside.gguf", "url": "https://example.test/model.gguf",
                  "sizeBytes": 1,
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }]
              }]
            }
            """;
        await File.WriteAllTextAsync(path, json);
        try
        {
            await Assert.That(() => ModelPackageCatalog.Load(path)).Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    [Arguments(".")]
    [Arguments("..")]
    [Arguments(".hidden")]
    public async Task CatalogRejectsUnsafePackageDirectoryName(string packageId)
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-catalog-{Guid.NewGuid():N}.json");
        var json = $$"""
            {
              "schemaVersion": 1,
              "packages": [{
                "id": "{{packageId}}", "displayName": "Unsafe", "architecture": "qwen2",
                "kind": "text-generation", "format": "gguf", "precision": "Q8_0",
                "parameterCount": 1, "license": "MIT",
                "sourceUrl": "https://example.test/model",
                "revision": "1111111111111111111111111111111111111111",
                "sets": ["smoke"],
                "files": [{
                  "path": "model.gguf", "url": "https://example.test/model.gguf",
                  "sizeBytes": 1,
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }]
              }]
            }
            """;
        await File.WriteAllTextAsync(path, json);
        try
        {
            await Assert.That(() => ModelPackageCatalog.Load(path)).Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task CatalogRejectsPortableFileNameCollision()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-catalog-{Guid.NewGuid():N}.json");
        const string json = /*lang=json,strict*/ """
            {
              "schemaVersion": 1,
              "packages": [{
                "id": "portable", "displayName": "Portable", "architecture": "qwen2",
                "kind": "text-generation", "format": "gguf", "precision": "Q8_0",
                "parameterCount": 1, "license": "MIT",
                "sourceUrl": "https://example.test/model",
                "revision": "1111111111111111111111111111111111111111",
                "sets": ["smoke"],
                "files": [
                  {"path": "Model.bin", "url": "https://example.test/one", "sizeBytes": 1,
                   "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                  {"path": "model.bin", "url": "https://example.test/two", "sizeBytes": 1,
                   "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
                ]
              }]
            }
            """;
        await File.WriteAllTextAsync(path, json);
        try
        {
            await Assert.That(() => ModelPackageCatalog.Load(path)).Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Synapse.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the Synapse repository root.");
    }
}
