using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Cli.Features.LayerDrop;

/// <summary>
/// Layer drop options (ADR-019): <c>--drop-layers i,j</c> is an experimental run, and <c>--drop-profile file.json</c>
/// takes its layers from an evaluation evidence file (<c>qualifiedDrop.layers</c>) whose SHA-256 qualifies it.
/// </summary>
internal static class LayerDropArguments
{
    public const string Usage = "[--drop-layers <i,j,...> | --drop-profile <evidence.json>]";

    /// <summary>False only for malformed input; no drop option yields a null profile.</summary>
    public static bool TryParse(IReadOnlyDictionary<string, string> values, out LayerDropProfile? drop)
    {
        drop = null;
        var hasLayers = values.TryGetValue("--drop-layers", out var layersText);
        var hasProfile = values.TryGetValue("--drop-profile", out var profilePath);
        if (hasLayers && hasProfile)
        {
            return false;
        }

        if (hasLayers)
        {
            var layers = ParseLayers(layersText!);
            drop = layers is null ? null : new LayerDropProfile(layers);
            return drop is not null;
        }

        return !hasProfile || TryReadProfile(profilePath!, out drop);
    }

    /// <summary>One stderr line naming the drop and, when qualified, its evidence digest.</summary>
    public static string Describe(LayerDropProfile drop) =>
        $"layer drop: layers {string.Join(',', drop.SortedLayers)}, " +
        (drop.EvidenceSha256 is { } evidence ? $"qualified by evidence sha256 {evidence}" : "experimental (no evidence)");

    private static int[]? ParseLayers(string text)
    {
        var parts = text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var layers = new int[parts.Length];
        for (var index = 0; index < parts.Length; index++)
        {
            if (!int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out layers[index]))
            {
                return null;
            }
        }

        return layers.Length > 0 ? layers : null;
    }

    private static bool TryReadProfile(string path, out LayerDropProfile? drop)
    {
        drop = null;
        if (!File.Exists(path))
        {
            return false;
        }

        var bytes = File.ReadAllBytes(path);
        using var json = JsonDocument.Parse(bytes);
        if (!json.RootElement.TryGetProperty("qualifiedDrop", out var qualified) ||
            !qualified.TryGetProperty("layers", out var layers) || layers.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        drop = new LayerDropProfile([.. layers.EnumerateArray().Select(layer => layer.GetInt32())])
        {
            EvidenceSha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)),
        };
        return true;
    }
}
