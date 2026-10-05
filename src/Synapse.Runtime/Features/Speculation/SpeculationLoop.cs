using System.Diagnostics;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.Speculation;

/// <summary>
/// One speculative generation on two direct slots (ADR-020). Slots are position-indexed and attention never reads
/// past the current position, so rejected drafts leave stale K/V only beyond the committed length, and the next
/// pass overwrites it before any read: that is the working branch's rollback.
/// </summary>
internal sealed class SpeculationLoop(
    IDecoderExecutor target,
    IBatchDecoder verifier,
    IDecoderExecutor draft,
    int draftVocabulary,
    int endOfSequenceToken)
{
    private readonly List<int> _sequence = [];
    private int _draftLength;
    private bool _draftable = true;

    public SpeculativeResult Run(IReadOnlyList<int> prompt, int maximumNewTokens, int draftTokens, bool adaptiveDepth)
    {
        var policy = adaptiveDepth ? new AdaptiveDepth(draftTokens) : null;
        var timer = Stopwatch.StartNew();
        target.Reserve(prompt.Count + maximumNewTokens + draftTokens);
        draft.Reserve(prompt.Count + maximumNewTokens + draftTokens);
        _sequence.AddRange(prompt);
        var first = GreedySampling.ArgMax(target.Prefill(prompt).Span);
        var timeToFirstToken = timer.Elapsed;
        _ = draft.Prefill(prompt);
        _draftLength = prompt.Count;
        Commit(first);
        var committed = 1;
        var (passes, drafted, accepted) = (0, 0, 0);
        while (committed < maximumNewTokens && _sequence[^1] != endOfSequenceToken)
        {
            var available = Math.Min(draftTokens, maximumNewTokens - committed - 1);
            var depth = _draftable ? policy?.Select(available) ?? available : 0;
            var started = policy is null ? 0 : Stopwatch.GetTimestamp();
            var proposals = Propose(depth);
            var proposed = policy is null ? 0 : Stopwatch.GetTimestamp();
            var (taken, added) = Verify(proposals, maximumNewTokens - committed);
            policy?.Observe(depth, taken, added,
                Stopwatch.GetElapsedTime(started, proposed).TotalMilliseconds,
                Stopwatch.GetElapsedTime(proposed).TotalMilliseconds,
                learn: available == draftTokens && _sequence[^1] != endOfSequenceToken);
            passes++;
            drafted += proposals.Length;
            accepted += taken;
            committed += added;
        }

        var generated = _sequence.Skip(prompt.Count).ToArray();
        return new SpeculativeResult(
            new TextGenerationResult([.. prompt], generated, timeToFirstToken, timer.Elapsed), passes, drafted, accepted)
        {
            DepthMeasurements = policy?.Measurements ?? [],
        };
    }

    /// <summary>The draft catches up on committed tokens it has not seen, then proposes <paramref name="count"/> tokens.</summary>
    private int[] Propose(int count)
    {
        if (count == 0)
        {
            return [];
        }

        ReadOnlyMemory<float> logits = default;
        for (; _draftLength < _sequence.Count; _draftLength++)
        {
            logits = draft.Decode(_sequence[_draftLength], _draftLength);
        }

        var proposals = new int[count];
        for (var index = 0; index < count; index++)
        {
            proposals[index] = GreedySampling.ArgMax(logits.Span);
            if (index + 1 < count)
            {
                logits = draft.Decode(proposals[index], _sequence.Count + index);
            }
        }

        // The draft holds K/V for the committed sequence plus every proposal but the last.
        _draftLength = _sequence.Count + count - 1;
        return proposals;
    }

    /// <summary>
    /// Evaluates the last committed token and the proposals in one target step, commits the longest agreeing prefix
    /// plus the target's next arg-max, and returns the accepted proposals and the committed count.
    /// </summary>
    private (int Accepted, int Committed) Verify(int[] proposals, int remaining)
    {
        var position = _sequence.Count - 1;
        if (proposals.Length == 0)
        {
            Commit(GreedySampling.ArgMax(target.Decode(_sequence[^1], position).Span));
            return (0, 1);
        }

        var step = new BatchToken[proposals.Length + 1];
        step[0] = new BatchToken(0, _sequence[^1], position, 0);
        for (var index = 0; index < proposals.Length; index++)
        {
            step[index + 1] = new BatchToken(0, proposals[index], position + index + 1, index + 1);
        }

        verifier.Forward(step, promptStart: step.Length);
        var start = _sequence.Count;
        var accepted = 0;
        for (var row = 0; row <= proposals.Length && _sequence.Count - start < remaining; row++)
        {
            var predicted = GreedySampling.ArgMax(verifier.GetLogits(row));
            Commit(predicted);
            if (row == proposals.Length || predicted != proposals[row])
            {
                break;
            }

            accepted++;
            if (predicted == endOfSequenceToken)
            {
                break;
            }
        }

        // The draft's K/V is valid only up to the first proposal the target rejected.
        _draftLength = Math.Min(_draftLength, start + accepted);
        return (accepted, _sequence.Count - start);
    }

    private void Commit(int token)
    {
        _sequence.Add(token);
        _draftable &= (uint)token < (uint)draftVocabulary;
    }
}
