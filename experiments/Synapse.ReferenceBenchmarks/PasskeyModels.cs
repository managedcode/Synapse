using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

internal sealed record PasskeyArguments(
    string SynapseExecutable,
    string ModelPath,
    string Scenario,
    int[] PromptTokens,
    double[] Depths,
    string Backend,
    int ContextSize,
    string Output,
    string? RopeScaling,
    int Threads,
    int MaxTokens,
    string KvPrecision)
{
    public static PasskeyArguments? Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index + 1 < args.Length; index += 2)
        {
            values[args[index]] = args[index + 1];
        }

        string[] required = ["--synapse-executable", "--model", "--scenario", "--prompt-tokens", "--depths", "--backend", "--context-size", "--output"];
        if (args.Length % 2 != 0 || required.Any(key => !values.ContainsKey(key)) ||
            !File.Exists(values["--synapse-executable"]) || !File.Exists(values["--model"]) ||
            !File.Exists(values["--scenario"]) ||
            !int.TryParse(values["--context-size"], NumberStyles.None, CultureInfo.InvariantCulture, out var context))
        {
            return null;
        }

        var prompts = Numbers(values["--prompt-tokens"], text => int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var n) ? n : -1);
        var depths = Numbers(values["--depths"], text => double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : -1);
        var threads = int.Parse(values.GetValueOrDefault("--threads", "8"), CultureInfo.InvariantCulture);
        var maxTokens = int.Parse(values.GetValueOrDefault("--max-tokens", "12"), CultureInfo.InvariantCulture);
        return prompts.Length > 0 && prompts.All(n => n > 0 && n + maxTokens <= context) &&
            depths.Length > 0 && depths.All(d => d is >= 0 and <= 1)
            ? new PasskeyArguments(
                values["--synapse-executable"], values["--model"], values["--scenario"], prompts, depths,
                values["--backend"], context, values["--output"], values.GetValueOrDefault("--rope-scaling"),
                threads, maxTokens, values.GetValueOrDefault("--kv-precision", "f32"))
            : null;
    }

    private static T[] Numbers<T>(string text, Func<string, T> parse) =>
        [.. text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(parse)];
}

internal sealed record PasskeyScenario(string Id, PasskeySegments Segments, IReadOnlyList<PasskeyKey> Keys);

internal sealed record PasskeySegments(int[] Prefix, int[] Filler, int[] Suffix);

internal sealed record PasskeyKey(string Key, int[] Needle, int[] Answer);

internal sealed record PasskeyRun(
    int PromptTokens,
    double Depth,
    string Key,
    IReadOnlyList<int> Answer,
    IReadOnlyList<int> GeneratedTokens,
    bool ExactAnswer,
    bool ContainsAnswer,
    int ExitCode,
    string? Subject,
    string? KernelImplementation,
    double? TimeToFirstTokenMilliseconds,
    double? PromptTokensPerSecondThroughFirstToken,
    double? DecodeTokensPerSecond,
    long? PeakPhysicalFootprintBytes,
    long? MaximumObservedWorkingSetBytes,
    double WallMilliseconds,
    string? Error)
{
    public static PasskeyRun From(
        int promptTokens,
        double depth,
        PasskeyKey key,
        int exitCode,
        string output,
        string error,
        ObservedProcessMetrics metrics,
        TimeSpan wall)
    {
        int[] generated = [];
        string? subject = null;
        string? kernel = null;
        double? ttft = null;
        double? prefill = null;
        double? decode = null;
        if (exitCode == 0)
        {
            using var json = JsonDocument.Parse(output);
            var root = json.RootElement;
            generated = [.. root.GetProperty("generated_tokens").EnumerateArray().Select(token => token.GetInt32())];
            subject = root.GetProperty("subject").GetString();
            kernel = root.GetProperty("kernel_implementation").GetString();
            ttft = root.GetProperty("time_to_first_token_milliseconds").GetDouble();
            prefill = Optional(root, "prompt_tokens_per_second_through_first_token");
            decode = Optional(root, "decode_tokens_per_second");
        }

        var exact = generated.AsSpan().StartsWith(key.Answer);
        var contains = Enumerable.Range(0, Math.Max(0, generated.Length - key.Answer.Length + 1))
            .Any(start => generated.AsSpan(start, key.Answer.Length).SequenceEqual(key.Answer));
        return new PasskeyRun(
            promptTokens, depth, key.Key, key.Answer, generated, exact, contains, exitCode, subject, kernel, ttft,
            prefill, decode, metrics.PeakPhysicalFootprintBytes, metrics.MaximumObservedWorkingSetBytes,
            wall.TotalMilliseconds, exitCode == 0 ? null : error[^Math.Min(error.Length, 2000)..]);
    }

    private static double? Optional(JsonElement root, string name) =>
        root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetDouble() : null;
}

internal sealed record PasskeyEvidence(
    int SchemaVersion,
    string Kind,
    DateTimeOffset RecordedAtUtc,
    string OperatingSystem,
    string ProcessArchitecture,
    int LogicalProcessors,
    string Scenario,
    string ModelPath,
    string Backend,
    int ContextSize,
    string? RopeScaling,
    int Threads,
    string KvPrecision,
    IReadOnlyList<PasskeyRun> Runs);

[JsonSerializable(typeof(PasskeyScenario))]
[JsonSerializable(typeof(PasskeyEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true)]
internal sealed partial class PasskeyJsonContext : JsonSerializerContext;
