using System.Security.Cryptography;
using System.Text;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>
/// Qualified layer drop (ADR-019): the listed source layers do not run, their residual passes through, and their
/// weights and KV are never touched. An approximation profile: with <see cref="EvidenceSha256"/> it is
/// quality-bounded by that evaluation; without it the run is experimental (the measurement that produces evidence).
/// </summary>
/// <param name="Layers">Source layer indices to drop, in any order.</param>
public sealed record LayerDropProfile(IReadOnlyList<int> Layers)
{
    /// <summary>Lower-case SHA-256 of the evaluation evidence file that qualified this drop set, or null.</summary>
    public string? EvidenceSha256 { get; init; }

    /// <summary>The dropped layers, sorted.</summary>
    public IReadOnlyList<int> SortedLayers => [.. Layers.Order()];

    /// <summary>Runtime-profile suffix, for example <c>drop4</c>, or <c>drop4x</c> when experimental.</summary>
    public string Name => $"drop{Layers.Count}" + (EvidenceSha256 is null ? "x" : string.Empty);

    /// <summary>
    /// The graph provenance key: the evidence digest, or for experimental runs the SHA-256 of the canonical request,
    /// so every drop set has its own graph fingerprint.
    /// </summary>
    public string ProvenanceSha256 => EvidenceSha256 ?? Convert.ToHexStringLower(
        SHA256.HashData(Encoding.UTF8.GetBytes("experimental-layer-drop:" + string.Join(',', SortedLayers))));

    /// <summary>Rejects empty, duplicate, or out-of-range layers, dropping every layer, and malformed evidence.</summary>
    public void Validate(int layerCount)
    {
        if (Layers.Count == 0 || Layers.Distinct().Count() != Layers.Count)
        {
            throw new ArgumentException("A layer drop lists one or more distinct layers.");
        }

        foreach (var layer in Layers)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(layer, nameof(Layers));
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, layerCount, nameof(Layers));
        }

        if (Layers.Count >= layerCount)
        {
            throw new ArgumentException("At least one layer must stay.");
        }

        if (EvidenceSha256 is { } evidence &&
            (evidence.Length != 64 || !evidence.All(character => character is (>= '0' and <= '9') or (>= 'a' and <= 'f'))))
        {
            throw new ArgumentException("Evidence is a lower-case 64-character SHA-256 digest.");
        }
    }

    /// <summary>Source indices of the layers that run, in order.</summary>
    public int[] KeptLayers(int layerCount) => [.. Enumerable.Range(0, layerCount).Except(Layers)];
}
