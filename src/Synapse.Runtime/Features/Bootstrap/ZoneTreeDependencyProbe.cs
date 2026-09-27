using ZoneTree;

namespace ManagedCode.Synapse.Runtime.Features.Bootstrap;

/// <summary>Exercises the required ZoneTree persistence dependency end to end.</summary>
public static class ZoneTreeDependencyProbe
{
    private const string ProbeKey = "synapse/bootstrap/durable-probe";

    /// <summary>Writes, closes, reopens, and reads a unique probe value.</summary>
    /// <param name="dataDirectory">Dedicated directory owned by the caller.</param>
    /// <returns><see langword="true"/> only when the reopened tree returns the exact value.</returns>
    public static bool VerifyDurableRoundTrip(string dataDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataDirectory);
        _ = Directory.CreateDirectory(dataDirectory);

        var expectedValue = Guid.NewGuid().ToString("N");
        using (var tree = Open(dataDirectory))
        using (tree.CreateMaintainer())
        {
            _ = tree.Upsert(ProbeKey, expectedValue);
        }

        using var reopenedTree = Open(dataDirectory);
        using var maintainer = reopenedTree.CreateMaintainer();
        return reopenedTree.TryGet(ProbeKey, out var actualValue)
            && StringComparer.Ordinal.Equals(expectedValue, actualValue);
    }

    private static IZoneTree<string, string> Open(string dataDirectory) =>
        new ZoneTreeFactory<string, string>()
            .SetDataDirectory(dataDirectory)
            .OpenOrCreate();
}
