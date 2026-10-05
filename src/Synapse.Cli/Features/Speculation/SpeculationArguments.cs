using System.Globalization;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.Speculation;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Cli.Features.Speculation;

/// <summary>What a speculative run reports next to the generated tokens (ADR-020).</summary>
internal sealed record SpeculationOutput(
    string DraftModel,
    string DraftProfile,
    int DraftTokens,
    int TargetPasses,
    int DraftedTokens,
    int AcceptedTokens,
    double AcceptanceRate,
    double CommittedTokensPerTargetPass,
    bool AdaptiveDepth,
    IReadOnlyList<SpeculativeDepthMeasurement> DepthMeasurements);

/// <summary>
/// Speculative decoding options (ADR-020): <c>--draft-model</c> names an external draft; <c>--draft-drop-layers</c>
/// drafts with a layer-dropped copy (ADR-019) of the draft file, or of the target file when no draft model is given, so
/// the draft's weights are pages the target already maps. The target verifies every token either way.
/// </summary>
internal sealed record SpeculationArguments(string? DraftModel, LayerDropProfile? DraftDrop, int DraftTokens, bool AdaptiveDepth)
{
    public const string Usage = "[--draft-model <model.synapse>] [--draft-drop-layers <i,j,...>] [--draft-tokens <1..7|auto>]";

    /// <summary>False only for malformed input; no draft option yields null.</summary>
    public static bool TryParse(IReadOnlyDictionary<string, string> values, out SpeculationArguments? speculation)
    {
        speculation = null;
        var model = values.GetValueOrDefault("--draft-model");
        LayerDropProfile? drop = null;
        if (values.TryGetValue("--draft-drop-layers", out var layers))
        {
            var parsed = layers.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(text => int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var layer) ? layer : -1)
                .ToArray();
            if (parsed.Length == 0 || parsed.Contains(-1))
            {
                return false;
            }

            drop = new LayerDropProfile(parsed);
        }

        var adaptive = values.GetValueOrDefault("--draft-tokens") == "auto";
        var tokens = adaptive ? 3 : GenerationOptions.ParsePositive(values.GetValueOrDefault("--draft-tokens"), 3);
        if (model is null && drop is null)
        {
            return !values.ContainsKey("--draft-tokens");
        }

        speculation = tokens is >= 1 and <= 7 ? new SpeculationArguments(model, drop, tokens, adaptive) : null;
        return speculation is not null;
    }

    /// <summary>Loads the draft next to <paramref name="target"/> and runs the request speculatively.</summary>
    public (TextGenerationResult Result, SpeculationOutput Output) Run(ITextGenerationModel target, GenerationOptions options)
    {
        var draftPath = DraftModel ?? options.ModelPath;
        using var draft = (Qwen2Model)ModelLoader.Load(draftPath, new ModelLoadOptions
        {
            ContextSize = options.ContextSize,
            MaximumParallelism = options.Threads,
            KernelBackend = options.Backend,
            MaximumConcurrentSessions = 1,
            ScoringRowsPerStep = 1,
            RopeScaling = options.RopeScaling,
            KvCachePrecision = options.KvCachePrecision,
            LayerDrop = DraftDrop,
        });
        var run = SpeculativeDecoding.Generate(
            target as Qwen2Model ?? throw new NotSupportedException("Speculative decoding runs Qwen2 targets."),
            draft,
            options.Tokens,
            options.MaximumTokens,
            DraftTokens,
            AdaptiveDepth);
        return (run.Result, new SpeculationOutput(
            Path.GetFullPath(draftPath),
            draft.RuntimeProfile,
            DraftTokens,
            run.TargetPasses,
            run.DraftedTokens,
            run.AcceptedTokens,
            run.DraftedTokens == 0 ? 0 : run.AcceptedTokens / (double)run.DraftedTokens,
            run.TargetPasses == 0 ? 0 : (run.Result.GeneratedTokens.Count - 1) / (double)run.TargetPasses,
            AdaptiveDepth,
            run.DepthMeasurements));
    }
}
