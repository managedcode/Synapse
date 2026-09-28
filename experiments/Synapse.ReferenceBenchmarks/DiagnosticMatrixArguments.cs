using System.Globalization;

internal sealed record DiagnosticMatrixArguments(
    string ModelPath,
    string Prompt,
    int[] PromptTokenIds,
    int[] ExpectedTokenIds,
    string ExpectedText,
    string SynapseExecutable,
    string SynapseBackend,
    string DotLlmExecutable,
    string DotLlmVersion,
    string LlamaCppExecutable,
    string LlamaCppVersion,
    int MaxTokens,
    int Threads,
    int Warmups,
    int Measurements,
    string OutputPath)
{
    public static DiagnosticMatrixArguments? Parse(string[] arguments)
    {
        if (arguments.Length == 0 || arguments.Length % 2 != 0)
        {
            return null;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < arguments.Length; index += 2)
        {
            if (!arguments[index].StartsWith("--", StringComparison.Ordinal) ||
                !values.TryAdd(arguments[index], arguments[index + 1]))
            {
                return null;
            }
        }

        var model = Required(values, "--model");
        var prompt = Required(values, "--prompt");
        var expectedText = Required(values, "--expected-text");
        var synapse = Required(values, "--synapse-executable");
        var synapseBackend = values.GetValueOrDefault("--synapse-backend", "managed");
        var dotLlm = Required(values, "--dotllm-executable");
        var dotLlmVersion = Required(values, "--dotllm-version");
        var llamaCpp = Required(values, "--llamacpp-executable");
        var llamaCppVersion = Required(values, "--llamacpp-version");
        var output = Required(values, "--output");
        var promptIds = ParseIds(Required(values, "--prompt-token-ids"));
        var expectedIds = ParseIds(Required(values, "--expected-token-ids"));
        var maxTokens = ParseCount(values, "--max-tokens", 8);
        var threads = ParseCount(values, "--threads", Environment.ProcessorCount);
        var warmups = ParseCount(values, "--warmups", 3);
        var measurements = ParseCount(values, "--measurements", 5);
        if (new[] { model, prompt, expectedText, synapse, dotLlm, dotLlmVersion,
            llamaCpp, llamaCppVersion, output }.Any(string.IsNullOrWhiteSpace) ||
            promptIds is not { Length: > 0 } || expectedIds is not { Length: > 0 } ||
            maxTokens is <= 0 or > 512 || threads is <= 0 or > 256 ||
            warmups is < 0 or > 10 || measurements is <= 0 or > 200 ||
            expectedIds.Length != maxTokens ||
            synapseBackend is not ("reference" or "managed" or "native") ||
            !new[] { model, synapse, dotLlm, llamaCpp }.All(File.Exists))
        {
            return null;
        }

        return new DiagnosticMatrixArguments(
            model!, prompt!, promptIds, expectedIds, expectedText!, synapse!, synapseBackend,
            dotLlm!, dotLlmVersion!, llamaCpp!, llamaCppVersion!, maxTokens,
            threads, warmups, measurements, output!);
    }

    private static string? Required(Dictionary<string, string> values, string name) =>
        values.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value)
            ? value
            : null;

    private static int[]? ParseIds(string? value)
    {
        if (value is null)
        {
            return null;
        }

        var parts = value.Split(',', StringSplitOptions.TrimEntries);
        var ids = new int[parts.Length];
        for (var index = 0; index < parts.Length; index++)
        {
            if (!int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out ids[index]))
            {
                return null;
            }
        }

        return ids;
    }

    private static int ParseCount(Dictionary<string, string> values, string name, int fallback) =>
        !values.TryGetValue(name, out var value) ? fallback :
        int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var count) ? count : -1;
}
