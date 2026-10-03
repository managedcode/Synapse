using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// Context sweep: every engine and KV cache type at several context sizes, measured on three axes (ADR-005,
/// ADR-015). Tokens: the answer check and agreement with a reference engine. Memory: weights, the KV cache the
/// context needs, and the measured peak footprint. Speed: time to first token, full generation, and decode rate.
/// Engines run one at a time with the order rotated each round. This is a diagnostic, not a paired verdict.
/// </summary>
internal static class SweepCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var options = SweepOptions.Parse(args);
        if (options is null)
        {
            await Console.Error.WriteLineAsync(SweepOptions.Usage).ConfigureAwait(false);
            return 2;
        }

        var tokenizer = TextTokenizers.FromGguf(options.ModelPath);
        var factory = new QualityTaskFactory(tokenizer, await File.ReadAllTextAsync(options.Haystack).ConfigureAwait(false));
        var model = SweepModel.Describe(options.ModelPath);
        var subjects = CreateSubjects(options);
        var samples = new List<SweepSample>();
        try
        {
            foreach (var context in options.Contexts)
            {
                var prompt = factory.Create("summary", context - options.MaxTokens - 64, 0.5, options.Seed + context);
                if (options.PromptsDirectory is { } directory)
                {
                    _ = Directory.CreateDirectory(directory);
                    await File.WriteAllTextAsync(Path.Combine(directory, $"ctx{context}.ids"), string.Join('\n', prompt.Tokens) + "\n")
                        .ConfigureAwait(false);
                    await File.WriteAllTextAsync(Path.Combine(directory, $"ctx{context}.txt"), prompt.Prompt).ConfigureAwait(false);
                }

                var eligible = subjects.Where(subject => !subject.CpuOnly || context <= options.CpuMaxContext).ToArray();
                for (var round = 0; round < options.Warmups + options.Measurements; round++)
                {
                    for (var index = 0; index < eligible.Length; index++)
                    {
                        var subject = eligible[(round + index) % eligible.Length];
                        var run = await subject.RunAsync(prompt, context, CancellationToken.None).ConfigureAwait(false);
                        var sample = new SweepSample(context, subject.Name, subject.KvCache, round, round < options.Warmups,
                            prompt.Grade(run.Output).Passed, run);
                        samples.Add(sample);
                        await Console.Error.WriteLineAsync(Describe(sample)).ConfigureAwait(false);
                    }
                }

                await WriteEvidenceAsync(options, model, tokenizer, samples).ConfigureAwait(false);
            }
        }
        finally
        {
            foreach (var subject in subjects)
            {
                await subject.DisposeAsync().ConfigureAwait(false);
            }
        }

        Console.WriteLine(SweepReport.Markdown(SweepReport.Summarize(options.Reference, model, tokenizer, samples)));
        return samples.All(sample => sample.Run.ExitCode == 0) ? 0 : 1;
    }

    private static List<ISweepSubject> CreateSubjects(SweepOptions options)
    {
        var subjects = new List<ISweepSubject>();
        foreach (var spec in options.Subjects)
        {
            subjects.Add(spec.Split(':') switch
            {
                ["synapse", var backend, var kv and ("f32" or "f16")] => new SynapseSweepSubject(options, backend, kv),
                ["llamacpp", var device and ("metal" or "cpu"), var kv] => new LlamaSweepSubject(options, device, kv),
                ["mlx"] => new MlxSweepSubject(options),
                _ => throw new ArgumentException($"Unknown sweep subject '{spec}'."),
            });
        }

        return subjects;
    }

    private static async Task WriteEvidenceAsync(
        SweepOptions options,
        SweepModel model,
        ITextTokenizer tokenizer,
        IReadOnlyList<SweepSample> samples)
    {
        var rows = SweepReport.Summarize(options.Reference, model, tokenizer, samples);
        var evidence = new SweepEvidence(1, "context-sweep-diagnostic", DateTimeOffset.UtcNow,
            RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture.ToString(), Environment.ProcessorCount,
            Path.GetFullPath(options.ModelPath), model, Path.GetFullPath(options.Haystack), options.MaxTokens,
            options.Warmups, options.Measurements, options.Threads, options.Reference, samples, rows, SweepReport.Markdown(rows));
        await File.WriteAllTextAsync(options.Output, JsonSerializer.Serialize(evidence, SweepJsonContext.Default.SweepEvidence))
            .ConfigureAwait(false);
    }

    private static string Describe(SweepSample sample) => string.Create(CultureInfo.InvariantCulture,
        $"sweep ctx={sample.Context} {sample.Subject}/{sample.KvCache} round={sample.Round}{(sample.Warmup ? " warmup" : string.Empty)} " +
        $"exit={sample.Run.ExitCode} ttft={sample.Run.TimeToFirstTokenMilliseconds:F0}ms gen={sample.Run.GenerationMilliseconds:F0}ms " +
        $"decode={sample.Run.DecodeTokensPerSecond:F1}tok/s footprint={sample.Run.PeakFootprintBytes / 1048576.0:F0}MiB " +
        $"correct={sample.Correct}");
}

/// <summary>Weights and KV geometry of the measured model, read once through the public model API.</summary>
internal sealed record SweepModel(long WeightsBytes, int Layers, int KeyValueHeads, int HeadDimension)
{
    public static SweepModel Describe(string modelPath)
    {
        using var model = (Qwen2Model)ModelLoader.Load(PreparedBenchmarkModel.Require(modelPath),
            new ModelLoadOptions { ContextSize = 16, KernelBackend = KernelBackend.Reference, MaximumParallelism = 1 });
        return new SweepModel(new FileInfo(modelPath).Length, model.LayerCount, model.KeyValueHeads,
            model.HiddenSize / model.AttentionHeads);
    }

    /// <summary>Bytes one position of K and V occupies for a cache type; <see langword="null"/> when not published.</summary>
    public double? KvBytesPerToken(string cache) => cache switch
    {
        "f32" => 4.0,
        "f16" or "bf16" => 2.0,
        "q8_0" => 34.0 / 32,
        "q4_0" => 18.0 / 32,
        _ => null,
    } * 2 * Layers * KeyValueHeads * HeadDimension;
}

internal sealed record SweepSample(
    int Context,
    string Subject,
    string KvCache,
    int Round,
    bool Warmup,
    bool Correct,
    SweepRun Run);

internal sealed record SweepEvidence(
    int SchemaVersion,
    string Kind,
    DateTimeOffset RecordedAt,
    string OperatingSystem,
    string Architecture,
    int LogicalProcessors,
    string ModelPath,
    SweepModel Model,
    string HaystackPath,
    int MaxTokens,
    int Warmups,
    int Measurements,
    int Threads,
    string Reference,
    IReadOnlyList<SweepSample> Samples,
    IReadOnlyList<SweepRow> Summary,
    string SummaryMarkdown,
    IReadOnlyList<string>? Sources = null);

[JsonSerializable(typeof(SweepEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
internal sealed partial class SweepJsonContext : JsonSerializerContext;
