using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

/// <summary>
/// A resident SwiftLM (MLX) server with separately converted weights: a separate-weights cohort that never counts as
/// engine parity (ADR-015). It runs without <c>--ctx-size</c>, because that option selects a rotating (sliding
/// window) KV cache that would drop early context. Only the prompt token count can be checked for identity.
/// </summary>
internal sealed class MlxServer : IAsyncDisposable
{
    private readonly Task _pumps;

    private MlxServer(Process process, Task pumps, string model, int port)
    {
        Process = process;
        _pumps = pumps;
        Model = model;
        Port = port;
    }

    public Process Process { get; }

    public string Model { get; }

    public int Port { get; }

    public HttpClient Client { get; } = new() { Timeout = TimeSpan.FromMinutes(30) };

    public static async Task<MlxServer> StartAsync(string binary, string model, int port)
    {
        var start = new ProcessStartInfo(binary)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var argument in new[] { "--model", model, "--host", "127.0.0.1", "--port",
            port.ToString(CultureInfo.InvariantCulture), "--temp", "0", "--max-tokens", "512" })
        {
            start.ArgumentList.Add(argument);
        }

        var process = Process.Start(start) ?? throw new InvalidOperationException("SwiftLM did not start.");
        var ready = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var pumps = Task.WhenAll(PumpAsync(process.StandardOutput, ready), PumpAsync(process.StandardError, ready));
        var server = new MlxServer(process, pumps, model, port);
        try
        {
            await ready.Task.WaitAsync(TimeSpan.FromMinutes(3)).ConfigureAwait(false);
            await server.WaitUntilListeningAsync().ConfigureAwait(false);
            return server;
        }
        catch
        {
            // A server that never became ready would stay resident and skew later memory and speed samples.
            await server.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    /// <summary>The ready log line can precede the listening socket, so the model list is polled until it answers.</summary>
    private async Task WaitUntilListeningAsync()
    {
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (true)
        {
            try
            {
                using var probe = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                using var response = await Client.GetAsync(new Uri($"http://127.0.0.1:{Port}/v1/models"), probe.Token)
                    .ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    return;
                }
            }
            catch (Exception exception) when (exception is HttpRequestException or OperationCanceledException &&
                DateTime.UtcNow < deadline)
            {
            }

            if (DateTime.UtcNow >= deadline)
            {
                throw new TimeoutException($"SwiftLM on port {Port} did not answer within 30 seconds.");
            }

            await Task.Delay(100).ConfigureAwait(false);
        }
    }

    /// <summary>A chat request with the harness system prompt; streaming lets the client time the first token.</summary>
    public HttpRequestMessage ChatRequest(QualityCase quality, int maxTokens, bool stream)
    {
        var body = new JsonObject
        {
            ["model"] = Model,
            ["messages"] = new JsonArray(
                new JsonObject { ["role"] = "system", ["content"] = QualityTaskFactory.SystemPrompt },
                new JsonObject { ["role"] = "user", ["content"] = quality.UserMessage }),
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0,
            ["stream"] = stream,
        };
        if (stream)
        {
            body["stream_options"] = new JsonObject { ["include_usage"] = true };
        }

        return new HttpRequestMessage(HttpMethod.Post, new Uri($"http://127.0.0.1:{Port}/v1/chat/completions"))
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
    }

    public async ValueTask DisposeAsync()
    {
        Client.Dispose();
        if (!Process.HasExited)
        {
            Process.Kill(entireProcessTree: true);
        }

        await Process.WaitForExitAsync().ConfigureAwait(false);
        await _pumps.ConfigureAwait(false);
        Process.Dispose();
    }

    private static async Task PumpAsync(StreamReader reader, TaskCompletionSource ready)
    {
        while (await reader.ReadLineAsync().ConfigureAwait(false) is { } line)
        {
            if (line.Contains("Ready. Listening", StringComparison.Ordinal) ||
                line.Contains("\"event\":\"ready\"", StringComparison.Ordinal))
            {
                _ = ready.TrySetResult();
            }
        }

        _ = ready.TrySetException(new InvalidOperationException("SwiftLM exited before it was ready."));
    }
}

/// <summary>MLX answers for the quality command: one non-streaming chat request per case.</summary>
internal sealed class MlxQualitySubject(MlxServer server) : IQualitySubject
{
    public string Name => "mlx-swiftlm";

    public static async Task<MlxQualitySubject> StartAsync(QualityRunOptions options) =>
        new(await MlxServer.StartAsync(options.MlxBinary!, options.MlxModel!, options.MlxPort).ConfigureAwait(false));

    public async Task<SubjectAnswer> AnswerAsync(QualityCase quality, CancellationToken cancellationToken)
    {
        var timer = Stopwatch.StartNew();
        using var request = server.ChatRequest(quality, quality.MaxTokens, stream: false);
        using var response = await server.Client.SendAsync(request, cancellationToken).ConfigureAwait(false);
        var text = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            return new SubjectAnswer(string.Empty, null, null, null, timer.Elapsed.TotalMilliseconds,
                (int)response.StatusCode, SynapseQualitySubject.Tail(text));
        }

        using var json = JsonDocument.Parse(text);
        var root = json.RootElement;
        var answer = root.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? string.Empty;
        int? promptTokens = root.TryGetProperty("usage", out var usage) && usage.TryGetProperty("prompt_tokens", out var count)
            ? count.GetInt32()
            : null;
        return new SubjectAnswer(answer.Trim(), promptTokens, promptTokens is null ? null : promptTokens == quality.Tokens.Length,
            null, timer.Elapsed.TotalMilliseconds, 0, null);
    }

    public ValueTask DisposeAsync() => server.DisposeAsync();
}
