using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Cli.Features.LongContext;

/// <summary>Explicit execution controls and fail-closed contradictions (ADR-025).</summary>
internal static class RuntimeOptimizationArguments
{
    public const string Usage = "[--optimization off|dense|custom]";

    public static bool TryParseRuntime(IReadOnlyDictionary<string, string> values, out KernelBackend backend,
        out RopeScaling? scaling, out KvCachePrecision kvPrecision)
    {
        backend = KernelBackend.Managed;
        scaling = null;
        kvPrecision = KvCachePrecision.Fp32;
        if ((values.TryGetValue("--backend", out var name) && !KernelBackendNames.TryParse(name, out backend)) ||
            (values.TryGetValue("--rope-scaling", out var text) && !GenerationOptions.TryParseScaling(text, out scaling)))
        {
            return false;
        }

        switch (values.GetValueOrDefault("--kv-precision", "f32"))
        {
            case "f32":
                return true;
            case "f16":
                kvPrecision = KvCachePrecision.Fp16;
                return true;
            default:
                return false;
        }
    }

    public static bool TryResolve(IReadOnlyDictionary<string, string> values, ref KernelBackend backend,
        out string mode)
    {
        mode = values.GetValueOrDefault("--optimization", "custom");
        if (mode is not ("off" or "dense" or "custom"))
        {
            return false;
        }

        if (mode == "custom")
        {
            return true;
        }

        if (values.GetValueOrDefault("--kv-precision", "f32") != "f32" ||
            values.Keys.Any(key => key is "--kv-pages" or "--drop-layers" or "--drop-profile" or
                "--draft-model" or "--draft-drop-layers" or "--draft-tokens"))
        {
            return false;
        }

        if (mode == "off")
        {
            if (values.ContainsKey("--backend") && backend != KernelBackend.Reference)
            {
                return false;
            }

            backend = KernelBackend.Reference;
        }

        return true;
    }

    public static bool IsGenerationFlag(string flag) => IsRuntimeFlag(flag) || flag is
        "--model" or "--tokens" or "--tokens-file" or "--max-tokens" or "--concurrent-requests" or
        "--draft-model" or "--draft-drop-layers" or "--draft-tokens";

    public static bool IsScoreFlag(string flag) => IsRuntimeFlag(flag) || flag is
        "--model" or "--text-file" or "--tokens-file" or "--chunks" or "--scoring-rows" or
        "--scores-output" or "--first-scored";

    private static bool IsRuntimeFlag(string flag) => flag is "--backend" or "--threads" or "--context-size" or
        "--rope-scaling" or "--kv-precision" or "--kv-pages" or "--drop-layers" or "--drop-profile" or
        "--optimization";
}
