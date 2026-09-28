using System.Globalization;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>
/// Combines a file-declared rotary scaling profile with the caller's explicit option (ADR-013). A declared
/// profile is used as-is; an explicit option must match it; unknown methods fail before any allocation.
/// </summary>
internal static class RopeScalingResolver
{
    public static RopeScaling? Resolve(
        IReadOnlyDictionary<string, object> metadata,
        string architecture,
        RopeScaling? requested)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentException.ThrowIfNullOrWhiteSpace(architecture);
        var declared = ReadDeclared(metadata, architecture);
        if (declared is not null && requested is not null && declared != requested)
        {
            throw new NotSupportedException(
                $"The model declares RoPE scaling {Describe(declared)}, but the load options request {Describe(requested)}.");
        }

        return declared ?? requested;
    }

    private static RopeScaling? ReadDeclared(IReadOnlyDictionary<string, object> metadata, string architecture)
    {
        var prefix = architecture + ".rope.scaling.";
        if (!metadata.TryGetValue(prefix + "type", out var type))
        {
            return metadata.Keys.Any(key => key.StartsWith(prefix, StringComparison.Ordinal))
                ? throw new NotSupportedException($"The model declares '{prefix}*' keys without '{prefix}type'.")
                : null;
        }

        if (type is not string kind || !string.Equals(kind, "yarn", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"RoPE scaling type '{type}' is not supported; only 'yarn' is implemented.");
        }

        return RopeScaling.Yarn(
            Convert.ToSingle(Required(metadata, prefix + "factor"), CultureInfo.InvariantCulture),
            Convert.ToInt32(Required(metadata, prefix + "original_context_length"), CultureInfo.InvariantCulture));
    }

    private static object Required(IReadOnlyDictionary<string, object> metadata, string key) =>
        metadata.TryGetValue(key, out var value)
            ? value
            : throw new NotSupportedException($"The model declares YaRN scaling without '{key}'.");

    private static string Describe(RopeScaling scaling) =>
        $"{scaling.Name} over {scaling.OriginalContextLength.ToString(CultureInfo.InvariantCulture)} positions";
}
