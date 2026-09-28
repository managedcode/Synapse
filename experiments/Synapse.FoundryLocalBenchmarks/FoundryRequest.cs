using System.Diagnostics;
using System.Text;
using Microsoft.AI.Foundry.Local;

internal static class FoundryRequest
{
    /// <summary>
    /// Sends one locked-transcript turn through a new streaming session. A new session per
    /// request keeps earlier requests' turns (and any KV state) out of the measurement.
    /// </summary>
    public static async Task<FoundrySample> MeasureAsync(IModel model, IReadOnlyList<DialogueTurn> turns,
        FoundryRequestPlan plan, Process process, CancellationToken cancellationToken)
    {
        using var session = new ChatSession(model);
        _ = session.SetStreaming(true).SetOptions(new RequestOptions
        {
            Search = new SearchOptions
            {
                Temperature = 0,
                DoSample = false,
                MaxOutputTokens = plan.MaxTokens,
            },
        });
        using var request = BuildRequest(turns, plan);
        process.Refresh();
        var cpuBefore = process.TotalProcessorTime;
        var answer = new StringBuilder();
        var reasoning = new StringBuilder();
        var streamedItems = 0;
        double? firstToken = null;
        var timer = Stopwatch.StartNew();
        await using var stream = session.ProcessStreamingRequestAsync(request, cancellationToken);
        await foreach (var item in stream.ConfigureAwait(false))
        {
            if (item is not TextItem text)
            {
                continue;
            }

            firstToken ??= timer.Elapsed.TotalMilliseconds;
            streamedItems++;
            _ = (text.Type == TextItemType.Reasoning ? reasoning : answer).Append(text.Text);
        }

        using var response = await stream.FinalResponse.ConfigureAwait(false);
        timer.Stop();
        process.Refresh();
        var usage = response.GetUsage();
        var wall = timer.Elapsed.TotalMilliseconds;
        var first = firstToken ?? throw new InvalidDataException("Foundry Local streamed no text item.");
        var decodeSeconds = Math.Max(0.000_001, (wall - first) / 1000d);
        return new FoundrySample(plan.TurnIndex + 1, plan.Round, plan.Warmup, "unreviewed",
            usage.PromptTokens, usage.CompletionTokens, streamedItems, response.FinishReason.ToString(),
            answer.ToString(), reasoning.ToString(), first, wall,
            usage.CompletionTokens > 1 ? (usage.CompletionTokens - 1) / decodeSeconds : null,
            (process.TotalProcessorTime - cpuBefore).TotalMilliseconds,
            PositiveOrNull(Math.Max(process.WorkingSet64, process.PeakWorkingSet64)),
            MacProcessFootprint.TryReadBytes(process.Id));
    }

    private static Request BuildRequest(IReadOnlyList<DialogueTurn> turns, FoundryRequestPlan plan)
    {
        var request = new Request();
        var prepend = plan.SystemPromptMode == FoundryModel.PrependToFirstUser;
        if (!prepend)
        {
            _ = request.AddItem(MessageItem.System(turns[0].System, null), true);
        }

        for (var turn = 0; turn <= plan.TurnIndex; turn++)
        {
            var user = prepend && turn == 0 ? $"{turns[0].System}\n\n{turns[0].User}" : turns[turn].User;
            _ = request.AddItem(MessageItem.User(user, null), true);
            if (turn < plan.TurnIndex)
            {
                _ = request.AddItem(MessageItem.Assistant(turns[turn].LockedAssistant, null), true);
            }
        }

        return request;
    }

    private static long? PositiveOrNull(long value) => value > 0 ? value : null;
}

internal sealed record FoundryRequestPlan(int TurnIndex, int Round, bool Warmup, int MaxTokens,
    string SystemPromptMode);
