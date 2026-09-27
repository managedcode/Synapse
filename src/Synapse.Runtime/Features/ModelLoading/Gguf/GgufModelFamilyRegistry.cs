using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal static class GgufModelFamilyRegistry
{
    private static readonly IReadOnlyDictionary<string, IGgufModelFamilyAdapter> Adapters =
        new IGgufModelFamilyAdapter[]
        {
            new Qwen2GgufModelFamilyAdapter(),
        }.ToDictionary(adapter => adapter.Architecture, StringComparer.Ordinal);

    public static IReadOnlyList<string> SupportedArchitectures { get; } =
        Array.AsReadOnly(Adapters.Keys.Order(StringComparer.Ordinal).ToArray());

    public static IGgufModelFamilyAdapter Resolve(GgufFile file)
    {
        var architecture = file.GetRequiredString("general.architecture");
        return Adapters.TryGetValue(architecture, out var adapter)
            ? adapter
            : throw new NotSupportedException(
                $"GGUF architecture '{architecture}' is unsupported. " +
                $"Supported architectures: {string.Join(", ", SupportedArchitectures)}.");
    }
}
