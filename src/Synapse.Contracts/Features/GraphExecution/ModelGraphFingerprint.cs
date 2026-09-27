using System.Security.Cryptography;

namespace ManagedCode.Synapse.Contracts.Features.GraphExecution;

/// <summary>Canonical lower-case SHA-256 identity of a complete Model IR graph.</summary>
/// <param name="Value">Lower-case hexadecimal SHA-256 digest.</param>
public readonly record struct ModelGraphFingerprint(string Value)
{
    /// <summary>Computes the fingerprint from a versioned, deterministic binary encoding.</summary>
    public static ModelGraphFingerprint Compute(ModelGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        new ModelGraphCanonicalEncoder(new CanonicalHashWriter(hash)).Write(graph);
        return new ModelGraphFingerprint(Convert.ToHexStringLower(hash.GetHashAndReset()));
    }

    /// <inheritdoc />
    public override string ToString() => Value;
}
