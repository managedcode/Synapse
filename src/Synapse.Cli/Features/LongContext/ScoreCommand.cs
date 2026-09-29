using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Cli.Features.LayerDrop;
using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.Cli.Features.LongContext;

/// <summary>
/// <c>synapse score</c>: teacher-forced perplexity with the llama-perplexity protocol (ADR-015). The stream is cut into
/// <c>floor(n / n_ctx)</c> chunks; each chunk starts from an empty cache and scores positions <c>n_ctx/2 .. n_ctx-2</c>.
/// </summary>
internal static class ScoreCommand
{
    public static int Run(IReadOnlyList<string> arguments)
    {
        var options = ScoreOptions.Parse(arguments);
        if (options is null)
        {
            Console.Error.WriteLine(ScoreOptions.Usage);
            return 2;
        }

        try
        {
            var tokens = ReadTokens(options);
            var chunks = Math.Min(tokens.Count / options.ContextSize, options.Chunks ?? int.MaxValue);
            if (chunks == 0)
            {
                Console.Error.WriteLine($"Scoring needs at least {options.ContextSize} tokens; the input has {tokens.Count}.");
                return 1;
            }

            var loadTimer = Stopwatch.StartNew();
            if (options.LayerDrop is { } drop)
            {
                Console.Error.WriteLine(LayerDropArguments.Describe(drop));
            }

            using var model = ModelLoader.Load(options.ModelPath, new ModelLoadOptions
            {
                ContextSize = options.ContextSize,
                MaximumParallelism = options.Threads,
                KernelBackend = options.Backend,
                RopeScaling = options.RopeScaling,
                KvCachePrecision = options.KvCachePrecision,
                KvPageActivation = options.KvPages,
                LayerDrop = options.LayerDrop,
                MaximumConcurrentSessions = 1,
                ScoringRowsPerStep = options.ScoringRows,
            });
            loadTimer.Stop();
            var run = ScoreChunks(model, tokens, options.ContextSize, chunks, options.FirstScored ?? (options.ContextSize / 2));
            var output = CreateOutput(options, model, tokens, run, loadTimer.Elapsed);
            Console.WriteLine(JsonSerializer.Serialize(output, ScoreJsonContext.Default.ScoreOutput));
            if (options.ScoresOutput is { } path)
            {
                File.WriteAllText(path, JsonSerializer.Serialize(
                    new ScoreTrace(output.TokensSha256, output.FirstScoredPosition, run.Traces),
                    ScoreJsonContext.Default.ScoreTrace));
            }

            return 0;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    internal static string Sha256(IReadOnlyList<int> tokens)
    {
        var text = new StringBuilder(tokens.Count * 7);
        foreach (var token in tokens)
        {
            _ = text.Append(token.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())));
    }

    private static IReadOnlyList<int> ReadTokens(ScoreOptions options)
    {
        if (options.TextFile is { } textFile)
        {
            return TextTokenizers.FromGguf(options.ModelPath)
                .Encode(File.ReadAllText(textFile), options.ParseSpecialTokens);
        }

        return GenerationOptions.ParseTokens(File.ReadAllText(options.TokensFile!), [',', ' ', '\t', '\r', '\n'])
            ?? throw new FormatException("The tokens file must hold decimal token IDs.");
    }

    private static ScoreRun ScoreChunks(ITextGenerationModel model, IReadOnlyList<int> tokens, int context, int chunks, int first)
    {
        var run = new ScoreRun();
        for (var chunk = 0; chunk < chunks; chunk++)
        {
            var slice = tokens.Skip(chunk * context).Take(context).ToArray();
            var scores = model.Score(slice, first, new ScoreProgress(chunk + 1, chunks));
            var matches = 0;
            var sum = 0.0;
            for (var index = 0; index < scores.NegativeLogLikelihoods.Count; index++)
            {
                var loss = scores.NegativeLogLikelihoods[index];
                (sum, run.Squares) = (sum + loss, run.Squares + (loss * loss));
                matches += scores.GreedyTokens[index] == slice[scores.FirstScoredPosition + index + 1] ? 1 : 0;
            }

            run.Sum += sum;
            run.Count += scores.NegativeLogLikelihoods.Count;
            run.Matches += matches;
            run.ChunkMean.Add(sum / scores.NegativeLogLikelihoods.Count);
            run.Cumulative.Add(Math.Exp(run.Sum / run.Count));
            run.ChunkMilliseconds.Add(scores.Elapsed.TotalMilliseconds);
            run.Traces.Add(new ChunkTrace(chunk, scores.NegativeLogLikelihoods, scores.GreedyTokens));
            Console.Error.WriteLine(string.Create(
                CultureInfo.InvariantCulture,
                $"score: chunk {chunk + 1}/{chunks} ppl {run.Cumulative[^1]:F4} in {scores.Elapsed.TotalSeconds:F1} s"));
        }

        return run;
    }

    private static ScoreOutput CreateOutput(
        ScoreOptions options,
        ITextGenerationModel model,
        IReadOnlyList<int> tokens,
        ScoreRun run,
        TimeSpan load)
    {
        var mean = run.Sum / run.Count;
        var variance = (run.Squares / run.Count) - (mean * mean);
        var perplexity = Math.Exp(mean);
        return new ScoreOutput(
            $"synapse-{model.RuntimeProfile}",
            KernelBackendNames.ToName(options.Backend),
            model.KernelImplementation,
            Path.GetFullPath(options.ModelPath),
            options.ContextSize,
            options.RopeScaling?.Name,
            options.KvCachePrecision == KvCachePrecision.Fp16 ? "f16" : "f32",
            options.ScoringRows,
            options.ParseSpecialTokens,
            tokens.Count,
            Sha256(tokens),
            run.ChunkMean.Count,
            options.FirstScored ?? (options.ContextSize / 2),
            run.Count,
            run.Cumulative,
            run.ChunkMean,
            perplexity,
            variance > 0 && run.Count > 1 ? perplexity * Math.Sqrt(variance / (run.Count - 1)) : null,
            mean,
            (double)run.Matches / run.Count,
            load.TotalMilliseconds,
            run.ChunkMilliseconds,
            run.ChunkMilliseconds.Sum());
    }

    private sealed class ScoreRun
    {
        public double Sum { get; set; }

        public double Squares { get; set; }

        public int Count { get; set; }

        public int Matches { get; set; }

        public List<double> ChunkMean { get; } = [];

        public List<double> Cumulative { get; } = [];

        public List<double> ChunkMilliseconds { get; } = [];

        public List<ChunkTrace> Traces { get; } = [];
    }

    /// <summary>Scoring progress on standard error at most once per second.</summary>
    private sealed class ScoreProgress(int chunk, int chunks) : IProgress<GenerationProgress>
    {
        private TimeSpan? _last;

        public void Report(GenerationProgress value)
        {
            if (_last is { } last && value.Elapsed - last < TimeSpan.FromSeconds(1))
            {
                return;
            }

            _last = value.Elapsed;
            Console.Error.WriteLine(string.Create(
                CultureInfo.InvariantCulture,
                $"progress: chunk {chunk}/{chunks}, evaluated {value.EvaluatedPromptTokens}/{value.PromptTokens}, {value.Elapsed.TotalSeconds:F1} s"));
        }
    }
}

internal sealed record ScoreOutput(
    string Subject,
    string KernelBackend,
    string KernelImplementation,
    string ModelPath,
    int ContextSize,
    string? RopeScaling,
    string KvPrecision,
    int ScoringRowsPerStep,
    bool ParseSpecialTokens,
    int TokenCount,
    string TokensSha256,
    int Chunks,
    int FirstScoredPosition,
    int ScoredTokens,
    IReadOnlyList<double> CumulativePerplexity,
    IReadOnlyList<double> ChunkMeanNegativeLogLikelihood,
    double Perplexity,
    double? PerplexityError,
    double MeanNegativeLogLikelihood,
    double GreedyAccuracy,
    double LoadMilliseconds,
    IReadOnlyList<double> ChunkMilliseconds,
    double ScoringMilliseconds);

internal sealed record ChunkTrace(int Chunk, IReadOnlyList<double> NegativeLogLikelihoods, IReadOnlyList<int> GreedyTokens);

internal sealed record ScoreTrace(string TokensSha256, int FirstScoredPosition, IReadOnlyList<ChunkTrace> Chunks);

[JsonSerializable(typeof(ScoreOutput))]
[JsonSerializable(typeof(ScoreTrace))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class ScoreJsonContext : JsonSerializerContext;
