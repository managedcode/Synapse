using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// Long-context answer quality across engines (ADR-015): deterministic exact-answer tasks at several lengths and
/// depths, the same prompt for every subject, raw JSON evidence written after every case, and a pass matrix.
/// This is a diagnostic (ADR-005), not a benchmark.
/// </summary>
internal static class QualityCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        var options = QualityRunOptions.Parse(args);
        if (options is null)
        {
            await Console.Error.WriteLineAsync(QualityRunOptions.Usage).ConfigureAwait(false);
            return 2;
        }

        var tokenizer = TextTokenizers.FromGguf(options.ModelPath);
        var haystack = await File.ReadAllTextAsync(options.Haystack).ConfigureAwait(false);
        var factory = new QualityTaskFactory(tokenizer, haystack);
        var subjects = await CreateSubjectsAsync(options, tokenizer).ConfigureAwait(false);
        var results = new List<QualityCaseResult>();
        try
        {
            foreach (var quality in CreateCases(options, factory))
            {
                var answers = new List<QualitySubjectResult>();
                foreach (var subject in subjects)
                {
                    var answer = await subject.AnswerAsync(quality, CancellationToken.None).ConfigureAwait(false);
                    var (score, passed) = quality.Grade(answer.Output);
                    answers.Add(new QualitySubjectResult(subject.Name, answer.Output, score, passed, answer.PromptTokens,
                        answer.TokenIdsIdentical, answer.PromptMilliseconds, answer.WallMilliseconds, answer.ExitCode, answer.Error));
                    await Console.Error.WriteLineAsync(string.Create(CultureInfo.InvariantCulture,
                        $"quality {quality.Task} {quality.Tokens.Length} tok depth {quality.Depth:F2} {subject.Name}: " +
                        $"{(passed ? "PASS" : "fail")} \"{Flatten(answer.Output)}\" ({answer.WallMilliseconds / 1000:F1} s)"))
                        .ConfigureAwait(false);
                }

                results.Add(QualityCaseResult.From(quality, answers));
                await WriteEvidenceAsync(options, haystack, subjects, results).ConfigureAwait(false);
            }
        }
        finally
        {
            foreach (var subject in subjects)
            {
                await subject.DisposeAsync().ConfigureAwait(false);
            }
        }

        Console.WriteLine(Summarize([.. subjects.Select(subject => subject.Name)], results));
        return results.SelectMany(result => result.Results).All(result => result.ExitCode == 0) ? 0 : 1;
    }

    internal static IEnumerable<QualityCase> CreateCases(QualityRunOptions options, QualityTaskFactory factory)
    {
        for (var length = 0; length < options.Lengths.Count; length++)
        {
            for (var task = 0; task < options.Tasks.Count; task++)
            {
                var depths = options.Tasks[task] == "vartrack" ? [0.5] : options.Depths;
                for (var depth = 0; depth < depths.Count; depth++)
                {
                    var seed = options.Seed + (length * 1_000) + (task * 100) + depth;
                    yield return factory.Create(options.Tasks[task], options.Lengths[length], depths[depth], seed);
                }
            }
        }
    }

    internal static string Summarize(IReadOnlyList<string> subjects, IReadOnlyList<QualityCaseResult> results)
    {
        var text = new StringBuilder();
        _ = text.Append("| task | target tokens | ").AppendJoin(" | ", subjects).AppendLine(" |");
        _ = text.Append("|---|---:|").Append(string.Concat(subjects.Select(_ => "---:|"))).AppendLine();
        foreach (var group in results.GroupBy(result => (result.Task, result.TargetTokens)))
        {
            _ = text.Append(CultureInfo.InvariantCulture, $"| {group.Key.Task} | {group.Key.TargetTokens} |");
            foreach (var subject in subjects)
            {
                var runs = group.SelectMany(result => result.Results).Where(result => result.Subject == subject).ToArray();
                _ = text.Append(CultureInfo.InvariantCulture, $" {runs.Count(run => run.Passed)}/{runs.Length} |");
            }

            _ = text.AppendLine();
        }

        return text.ToString();
    }

    private static async Task<List<IQualitySubject>> CreateSubjectsAsync(QualityRunOptions options, ITextTokenizer tokenizer)
    {
        var subjects = new List<IQualitySubject>();
        foreach (var spec in options.Subjects)
        {
            subjects.Add(spec.Split(':') switch
            {
                ["synapse", var backend, var kv] => new SynapseQualitySubject(options, backend, kv, tokenizer),
                ["synapse", var backend, var kv, var variant] => new SynapseQualitySubject(options, backend, kv, tokenizer, variant),
                ["llamacpp", var device and ("metal" or "cpu")] => new LlamaQualitySubject(options, device),
                ["mlx"] => await MlxQualitySubject.StartAsync(options).ConfigureAwait(false),
                _ => throw new ArgumentException($"Unknown quality subject '{spec}'."),
            });
        }

        return subjects;
    }

    private static async Task WriteEvidenceAsync(
        QualityRunOptions options,
        string haystack,
        IReadOnlyList<IQualitySubject> subjects,
        IReadOnlyList<QualityCaseResult> results)
    {
        var evidence = new QualityEvidence(
            1,
            "long-context-quality-diagnostic",
            DateTimeOffset.UtcNow,
            RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            Environment.ProcessorCount,
            Path.GetFullPath(options.ModelPath),
            Path.GetFullPath(options.Haystack),
            Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(haystack))),
            options.ContextSize,
            options.RopeScaling,
            options.Seed,
            [.. subjects.Select(subject => subject.Name)],
            results,
            Summarize([.. subjects.Select(subject => subject.Name)], results));
        await File.WriteAllTextAsync(options.Output, JsonSerializer.Serialize(evidence, QualityJsonContext.Default.QualityEvidence))
            .ConfigureAwait(false);
    }

    private static string Flatten(string text)
    {
        var flat = text.ReplaceLineEndings(" ");
        return flat.Length <= 80 ? flat : flat[..80] + "…";
    }
}

internal sealed record QualitySubjectResult(
    string Subject,
    string Output,
    double Score,
    bool Passed,
    int? PromptTokens,
    bool? TokenIdsIdentical,
    double? PromptMilliseconds,
    double WallMilliseconds,
    int ExitCode,
    string? Error);

internal sealed record QualityCaseResult(
    string Task,
    int TargetTokens,
    int PromptTokens,
    double Depth,
    int Seed,
    string PromptTokenIdsSha256,
    IReadOnlyList<string> Answers,
    IReadOnlyList<QualitySubjectResult> Results)
{
    public static QualityCaseResult From(QualityCase quality, IReadOnlyList<QualitySubjectResult> results) => new(
        quality.Task,
        quality.TargetTokens,
        quality.Tokens.Length,
        quality.Depth,
        quality.Seed,
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(
            string.Concat(quality.Tokens.Select(id => id.ToString(CultureInfo.InvariantCulture) + "\n"))))),
        quality.Answers,
        results);
}

internal sealed record QualityEvidence(
    int SchemaVersion,
    string Kind,
    DateTimeOffset RecordedAt,
    string OperatingSystem,
    string Architecture,
    int LogicalProcessors,
    string ModelPath,
    string HaystackPath,
    string HaystackSha256,
    int ContextSize,
    string? RopeScaling,
    int Seed,
    IReadOnlyList<string> Subjects,
    IReadOnlyList<QualityCaseResult> Cases,
    string SummaryMarkdown);

[JsonSerializable(typeof(QualityEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
internal sealed partial class QualityJsonContext : JsonSerializerContext;
