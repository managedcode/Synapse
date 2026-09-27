using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;

namespace ManagedCode.Synapse.Cli.Features.ModelPackages;

internal static class ModelCommand
{
    public static async Task<int> RunAsync(IReadOnlyList<string> arguments)
    {
        if (arguments.Count == 0)
        {
            PrintUsage();
            return 2;
        }

        try
        {
            var options = ModelCommandOptions.Parse(arguments);
            if (options is null)
            {
                PrintUsage();
                return 2;
            }

            var catalog = ModelPackageCatalog.Load(options.CatalogPath);
            if (options.Operation == "list")
            {
                foreach (var package in catalog.Packages)
                {
                    Console.WriteLine($"{package.Id}\t{package.Architecture}\t{package.Kind}\t{package.Precision}");
                }

                return 0;
            }

            var packages = SelectPackages(catalog, options);
            using var handler = new HttpClientHandler { AllowAutoRedirect = false };
            using var client = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
            var downloader = new ModelPackageDownloader(client);
            var results = new List<DownloadedModelPackage>(packages.Count);
            foreach (var package in packages)
            {
                Console.Error.WriteLine($"Fetching {package.Id}...");
                results.Add(await downloader.FetchAsync(
                    package,
                    options.OutputRoot,
                    CancellationToken.None).ConfigureAwait(false));
            }

            Console.WriteLine(JsonSerializer.Serialize(results, ModelCommandJsonContext.Default.ListDownloadedModelPackage));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static void PrintUsage() => Console.Error.WriteLine(
        "Usage: synapse model list [--catalog models/catalog.json] | " +
        "synapse model fetch (--set <name> | --id <package>) " +
        "[--catalog models/catalog.json] [--output artifacts/models]");

    private static IReadOnlyList<ModelPackageDefinition> SelectPackages(
        ModelPackageCatalog catalog,
        ModelCommandOptions options) => options.PackageId is not null
            ? [catalog.GetRequired(options.PackageId)]
            : catalog.SelectSet(options.SetName!);
}

internal sealed record ModelCommandOptions(
    string Operation,
    string CatalogPath,
    string OutputRoot,
    string? SetName,
    string? PackageId)
{
    public static ModelCommandOptions? Parse(IReadOnlyList<string> arguments)
    {
        var operation = arguments[0];
        if (operation is not ("list" or "fetch"))
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 1; index < arguments.Count; index += 2)
        {
            if (index + 1 >= arguments.Count || !arguments[index].StartsWith("--", StringComparison.Ordinal))
            {
                return null;
            }

            values[arguments[index]] = arguments[index + 1];
        }

        if (values.Keys.Any(key => key is not ("--catalog" or "--output" or "--set" or "--id")))
        {
            return null;
        }

        var setName = values.GetValueOrDefault("--set");
        var packageId = values.GetValueOrDefault("--id");
        if (operation == "fetch" && (string.IsNullOrWhiteSpace(setName) == string.IsNullOrWhiteSpace(packageId)))
        {
            return null;
        }

        return new ModelCommandOptions(
            operation,
            values.GetValueOrDefault("--catalog", "models/catalog.json"),
            values.GetValueOrDefault("--output", "artifacts/models"),
            setName,
            packageId);
    }
}

[JsonSerializable(typeof(List<DownloadedModelPackage>))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class ModelCommandJsonContext : JsonSerializerContext;
