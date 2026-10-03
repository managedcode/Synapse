using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// <c>layer-drop</c>: qualifies layer drops (ADR-019) on pinned tokens. It scores the dense model, then every
/// single-layer drop, then cumulative drops of the cheapest layers, and reports perplexity and top-1 agreement with
/// the dense model at every scored position. All runs are experimental; the output file becomes the evidence that
/// qualifies a chosen drop set (<c>qualifiedDrop.layers</c>).
/// </summary>
internal static class LayerDropCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index + 1 < args.Length; index += 2)
        {
            values[args[index]] = args[index + 1];
        }

        if (!values.TryGetValue("--model", out var model) || !values.TryGetValue("--tokens-file", out var tokensFile) ||
            !values.TryGetValue("--output", out var output))
        {
            await Console.Error.WriteLineAsync("Usage: layer-drop --model <gguf> --tokens-file <ids> --output <file.json> " +
                "[--tokens 2048] [--first-scored 1024] [--cumulative 8] [--backend metal] [--kv-precision f16] " +
                "[--qualify <k>]").ConfigureAwait(false);
            return 2;
        }

        var limit = Number(values, "--tokens", 2_048);
        int[] tokens = [.. (await File.ReadAllTextAsync(tokensFile).ConfigureAwait(false))
            .Split(['\n', ',', ' '], StringSplitOptions.RemoveEmptyEntries).Select(text => int.Parse(text, CultureInfo.InvariantCulture))
            .Take(limit)];
        var first = Number(values, "--first-scored", tokens.Length / 2);
        var runner = new DropRunner(model, tokens, first, values);
        var dense = runner.Run(null);
        var layerCount = dense.LayerCount;
        var singles = Enumerable.Range(0, layerCount).Select(layer => runner.Run([layer], dense)).ToArray();
        var ranked = singles.OrderBy(run => run.Perplexity).Select(run => run.Layers[0]).ToArray();
        var cumulative = Enumerable.Range(2, Math.Max(0, Number(values, "--cumulative", 8) - 1))
            .Select(count => runner.Run([.. ranked.Take(count)], dense)).ToArray();
        var qualify = values.TryGetValue("--qualify", out var qualifyText) ? int.Parse(qualifyText, CultureInfo.InvariantCulture) : 0;
        var evidence = new LayerDropEvidence(1, "layer-drop-qualification", DateTimeOffset.UtcNow, Path.GetFullPath(model),
            Path.GetFullPath(tokensFile), tokens.Length, first, dense.RuntimeProfile, dense, singles, ranked, cumulative,
            qualify > 0 ? new QualifiedDrop([.. ranked.Take(qualify).Order()]) : null);
        await File.WriteAllTextAsync(output, JsonSerializer.Serialize(evidence, LayerDropJsonContext.Default.LayerDropEvidence))
            .ConfigureAwait(false);
        return 0;
    }

    private static int Number(Dictionary<string, string> values, string key, int fallback) =>
        values.TryGetValue(key, out var text) ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;

    private sealed class DropRunner(string model, int[] tokens, int first, Dictionary<string, string> values)
    {
        public DropRun Run(int[]? layers, DropRun? dense = null)
        {
            var prepared = PreparedBenchmarkModel.Require(model);
            var timer = Stopwatch.StartNew();
            using var loaded = (Qwen2Model)ModelLoader.Load(prepared, new ModelLoadOptions
            {
                ContextSize = tokens.Length,
                KernelBackend = KernelBackendNames.TryParse(values.GetValueOrDefault("--backend", "metal"), out var backend)
                    ? backend
                    : KernelBackend.Metal,
                KvCachePrecision = values.GetValueOrDefault("--kv-precision", "f16") == "f32" ? KvCachePrecision.Fp32 : KvCachePrecision.Fp16,
                ScoringRowsPerStep = 64,
                LayerDrop = layers is null ? null : new LayerDropProfile(layers),
            });
            var scores = loaded.Score(tokens, first, progress: null);
            var perplexity = Math.Exp(scores.NegativeLogLikelihoods.Average());
            var agreement = dense is null
                ? 1.0
                : scores.GreedyTokens.Zip(dense.GreedyTokens, (left, right) => left == right ? 1.0 : 0.0).Average();
            var run = new DropRun(layers ?? [], loaded.LayerCount + (layers?.Length ?? 0), loaded.RuntimeProfile, perplexity,
                dense is null ? 0 : ((perplexity / dense.Perplexity) - 1) * 100, agreement, scores.Elapsed.TotalSeconds,
                timer.Elapsed.TotalSeconds, [.. scores.GreedyTokens]);
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"layer-drop [{string.Join(',', run.Layers)}] ppl {run.Perplexity:F4} ({run.DeltaPercent:+0.00;-0.00}%) top-1 {run.TopOneAgreement:P1} in {run.TotalSeconds:F1} s"));
            return run;
        }
    }
}

internal sealed record DropRun(
    int[] Layers,
    int LayerCount,
    string RuntimeProfile,
    double Perplexity,
    double DeltaPercent,
    double TopOneAgreement,
    double ScoringSeconds,
    double TotalSeconds,
    [property: JsonIgnore] int[] GreedyTokens);

internal sealed record QualifiedDrop(int[] Layers);

internal sealed record LayerDropEvidence(
    int SchemaVersion,
    string Kind,
    DateTimeOffset RecordedAt,
    string ModelPath,
    string TokensFile,
    int TokenCount,
    int FirstScoredPosition,
    string DenseProfile,
    DropRun Dense,
    IReadOnlyList<DropRun> SingleLayer,
    IReadOnlyList<int> RankedLayers,
    IReadOnlyList<DropRun> Cumulative,
    QualifiedDrop? QualifiedDrop);

[JsonSerializable(typeof(LayerDropEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true)]
internal sealed partial class LayerDropJsonContext : JsonSerializerContext;
