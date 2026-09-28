using System.Diagnostics;
using System.Text;
using System.Text.Json;

/// <summary>
/// MLX through a resident SwiftLM server with separately converted weights (a separate-weights cohort). The client
/// times the first streamed token and the last one; memory is the server's peak footprint during the request.
/// </summary>
internal sealed class MlxSweepSubject(MlxServer server, int maxTokens) : ISweepSubject
{
    public string Name => "mlx-swiftlm";

    public string KvCache => "native";

    public bool CpuOnly => false;

    public async Task<SweepRun> RunAsync(QualityCase prompt, int context, CancellationToken cancellationToken)
    {
        await using var sampler = new ProcessMemorySampler(server.Process);
        var timer = Stopwatch.StartNew();
        using var request = server.ChatRequest(prompt, maxTokens, stream: true);
        using var response = await server.Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            var failure = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            return new SweepRun((int)response.StatusCode, string.Empty, null, null, null, null, 0, null,
                timer.Elapsed.TotalMilliseconds, null, null, SynapseQualitySubject.Tail(failure));
        }

        var (text, first, promptTokens, completionTokens) = await ReadStreamAsync(response, timer, cancellationToken)
            .ConfigureAwait(false);
        var total = timer.Elapsed.TotalMilliseconds;
        var metrics = await sampler.CompleteAsync().ConfigureAwait(false);
        var generated = completionTokens ?? 0;
        return new SweepRun(0, text.Trim(), promptTokens, promptTokens is null ? null : promptTokens == prompt.Tokens.Length,
            first, total, generated,
            first is { } ttft && generated > 1 && total > ttft ? (generated - 1) / ((total - ttft) / 1000) : null,
            total, metrics.PeakPhysicalFootprintBytes, metrics.MaximumObservedWorkingSetBytes, null);
    }

    public ValueTask DisposeAsync() => server.DisposeAsync();

    private static async Task<(string Text, double? First, int? PromptTokens, int? CompletionTokens)> ReadStreamAsync(
        HttpResponseMessage response,
        Stopwatch timer,
        CancellationToken cancellationToken)
    {
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream);
        var text = new StringBuilder();
        double? first = null;
        int? promptTokens = null;
        int? completionTokens = null;
        while (await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal) || line == "data: [DONE]")
            {
                continue;
            }

            using var chunk = JsonDocument.Parse(line[6..]);
            var root = chunk.RootElement;
            if (root.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object)
            {
                promptTokens = usage.TryGetProperty("prompt_tokens", out var p) ? p.GetInt32() : promptTokens;
                completionTokens = usage.TryGetProperty("completion_tokens", out var c) ? c.GetInt32() : completionTokens;
            }

            if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0 &&
                choices[0].GetProperty("delta").TryGetProperty("content", out var content) &&
                content.GetString() is { Length: > 0 } piece)
            {
                first ??= timer.Elapsed.TotalMilliseconds;
                _ = text.Append(piece);
            }
        }

        return (text.ToString(), first, promptTokens, completionTokens);
    }
}
