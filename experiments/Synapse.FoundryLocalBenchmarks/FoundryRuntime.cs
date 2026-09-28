using System.Reflection;
using System.Text.Json;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging.Abstractions;

internal static class FoundryRuntime
{
    public const string WebGpuProvider = "WebGpuExecutionProvider";

    /// <summary>Creates the in-process runtime with every file under one explicit directory.</summary>
    public static async Task<FoundryLocalManager> StartAsync(string cacheDirectory,
        CancellationToken cancellationToken)
    {
        var root = Path.GetFullPath(cacheDirectory);
        _ = Directory.CreateDirectory(root);
        await FoundryLocalManager.CreateAsync(new Configuration
        {
            AppName = "synapse-benchmark",
            AppDataDir = Path.Combine(root, ".app"),
            ModelCacheDir = root,
            LogsDir = Path.Combine(root, ".logs"),
            LogLevel = LogLevel.Warning,
            DisableNonessentialTelemetry = true,
        }, NullLogger.Instance, cancellationToken).ConfigureAwait(false);
        return FoundryLocalManager.Instance;
    }

    /// <summary>Resolves the exact pinned variant and rejects catalog drift.</summary>
    public static async Task<IModel> ResolveAsync(FoundryLocalManager manager, FoundryVariant pinned,
        CancellationToken cancellationToken)
    {
        var catalog = await manager.GetCatalogAsync(cancellationToken).ConfigureAwait(false);
        var variant = await catalog.GetModelVariantAsync(pinned.Id, cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidDataException($"The Foundry catalog has no variant '{pinned.Id}'.");
        var expectedDevice = pinned.Device == "gpu" ? DeviceType.GPU : DeviceType.CPU;
        if (variant.Info.FileSizeMb != pinned.FileSizeMb || variant.Info.Runtime?.DeviceType != expectedDevice)
        {
            throw new InvalidDataException(
                $"Catalog variant '{pinned.Id}' reports {variant.Info.FileSizeMb} MB on " +
                $"{variant.Info.Runtime?.DeviceType}; the set pins {pinned.FileSizeMb} MB on {expectedDevice}.");
        }

        return variant;
    }

    /// <summary>
    /// Registers the WebGPU plugin that GPU variants need. Registration lasts one process, so
    /// <c>run</c> re-registers a plugin that <c>fetch</c> already placed in the cache and never downloads it.
    /// </summary>
    public static async Task RequireGpuProviderAsync(FoundryLocalManager manager, string cacheDirectory,
        bool allowDownload, CancellationToken cancellationToken)
    {
        var provider = manager.DiscoverEps().FirstOrDefault(ep => ep.Name == WebGpuProvider)
            ?? throw new NotSupportedException($"{WebGpuProvider} is not available on this platform.");
        if (provider.IsRegistered)
        {
            return;
        }

        var bundles = Path.Combine(Path.GetFullPath(cacheDirectory), ".app", "ep", "webgpu-ep", "bundles");
        if (!allowDownload && (!Directory.Exists(bundles) || !Directory.EnumerateDirectories(bundles).Any()))
        {
            throw new FoundryNotCachedException($"{WebGpuProvider} is not in {bundles}; run fetch --device gpu first.");
        }

        var result = await manager.DownloadAndRegisterEpsAsync([WebGpuProvider], cancellationToken)
            .ConfigureAwait(false);
        if (!result.Success)
        {
            throw new InvalidOperationException($"{WebGpuProvider} registration failed: {result.Status}");
        }
    }

    /// <summary>Reads the resolved NuGet package version from this process's deps.json.</summary>
    public static string PackageVersion(string packageId)
    {
        var depsPath = Path.Combine(AppContext.BaseDirectory,
            $"{typeof(FoundryRuntime).Assembly.GetName().Name}.deps.json");
        using var deps = JsonDocument.Parse(File.ReadAllText(depsPath));
        var prefix = packageId + "/";
        return deps.RootElement.GetProperty("libraries").EnumerateObject()
            .Select(library => library.Name)
            .FirstOrDefault(name => name.StartsWith(prefix, StringComparison.Ordinal))?[prefix.Length..]
            ?? throw new InvalidDataException($"Package {packageId} is not in {depsPath}.");
    }

    public static string SdkAssemblyVersion() => InformationalVersion(typeof(FoundryLocalManager).Assembly);

    public static string NativeRuntimePath() =>
        Directory.EnumerateFiles(AppContext.BaseDirectory, "*foundry_local.*")
            .FirstOrDefault(path => Path.GetExtension(path) is ".dylib" or ".so" or ".dll")
        ?? throw new FileNotFoundException("The Foundry Local native runtime is missing from the output.");

    private static string InformationalVersion(Assembly assembly) =>
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? assembly.GetName().Version?.ToString() ?? "unknown";
}
