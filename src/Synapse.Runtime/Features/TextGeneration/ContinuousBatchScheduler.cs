using System.Collections.Concurrent;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// Continuous batching for one model instance (ADR-007). A dedicated thread is the only writer of scheduled KV
/// slots. Each step feeds one token per decoding session first, then first-in first-out prompt chunks, and
/// runs them in one forward pass under the model's execution gate.
/// </summary>
internal sealed class ContinuousBatchScheduler(IBatchDecoder decoder, object executionGate, int endOfSequenceToken)
    : IDisposable
{
    private readonly IBatchDecoder _decoder = decoder;
    private readonly object _executionGate = executionGate;
    private readonly int _endOfSequenceToken = endOfSequenceToken;
    private readonly ConcurrentQueue<BatchRequest> _inbox = new();
    private readonly SemaphoreSlim _signal = new(0);
    private readonly List<BatchRequest> _waiting = [];
    private readonly List<BatchRequest> _active = [];
    private readonly BatchToken[] _step = new BatchToken[decoder.StepTokenCapacity];
    private readonly Lock _startGate = new();
    private Thread? _loop;
    private volatile bool _disposed;

    public event Action<BatchStepTrace>? StepCompleted;

    public Task<TextGenerationResult> EnqueueAsync(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled<TextGenerationResult>(cancellationToken);
        }

        var request = new BatchRequest([.. promptTokens], maximumNewTokens, cancellationToken)
        {
            Registration = cancellationToken.Register(static state =>
            {
                var scheduler = (ContinuousBatchScheduler)state!;
                if (!scheduler._disposed)
                {
                    _ = scheduler._signal.Release();
                }
            }, this),
        };
        _inbox.Enqueue(request);
        EnsureStarted();
        _ = _signal.Release();
        return request.Completion.Task;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _ = _signal.Release();
        _loop?.Join();
        var abandoned = new ObjectDisposedException(nameof(ContinuousBatchScheduler));
        foreach (var request in _active.Concat(_waiting).Concat(_inbox))
        {
            request.Fail(abandoned);
        }

        _signal.Dispose();
    }

    private void EnsureStarted()
    {
        lock (_startGate)
        {
            _loop ??= StartLoop();
        }
    }

    private Thread StartLoop()
    {
        var thread = new Thread(Run) { IsBackground = true, Name = "synapse-batch-scheduler" };
        thread.Start();
        return thread;
    }

    private void Run()
    {
        while (!_disposed)
        {
            RetireCancelled();
            Admit();
            var count = Compose();
            if (count == 0)
            {
                _signal.Wait();
                continue;
            }

            try
            {
                lock (_executionGate)
                {
                    _decoder.Forward(_step.AsSpan(0, count));
                }
            }
            catch (Exception exception)
            {
                FailActive(exception);
                continue;
            }

            Advance();
        }
    }

    private void Admit()
    {
        while (_inbox.TryDequeue(out var request))
        {
            _waiting.Add(request);
        }

        while (_waiting.Count > 0 && _active.Count < _decoder.SessionSlots - 1)
        {
            var request = _waiting[0];
            _waiting.RemoveAt(0);
            request.Slot = FreeSlot();
            _active.Add(request);
        }
    }

    private int FreeSlot()
    {
        for (var slot = 1; slot < _decoder.SessionSlots; slot++)
        {
            if (_active.TrueForAll(request => request.Slot != slot))
            {
                return slot;
            }
        }

        throw new InvalidOperationException("No free KV slot although admission reported capacity.");
    }

    private int Compose()
    {
        var count = 0;
        var logitsRow = 0;
        foreach (var request in _active.Where(request => !request.InPrefill))
        {
            request.BeginStep(prefill: 0, decode: 1, logitsRow);
            _step[count++] = new BatchToken(request.Slot, request.Generated[^1], request.Length, logitsRow++);
        }

        foreach (var request in _active.Where(request => request.InPrefill))
        {
            var take = Math.Min(_step.Length - count, request.Prompt.Length - request.Length);
            var completes = take > 0 && request.Length + take == request.Prompt.Length;
            request.BeginStep(prefill: take, decode: 0, completes ? logitsRow : -1);
            for (var index = 0; index < take; index++)
            {
                var position = request.Length + index;
                var row = completes && index == take - 1 ? logitsRow++ : -1;
                _step[count++] = new BatchToken(request.Slot, request.Prompt[position], position, row);
            }
        }

        return count;
    }

    private void Advance()
    {
        var entries = _active
            .Where(request => request.StepTokens > 0)
            .Select(request => new BatchStepEntry(
                request.Slot,
                request.Prompt.Length,
                request.StepPrefill,
                request.StepDecode,
                request.Generated.Count))
            .ToArray();
        foreach (var request in _active.ToArray())
        {
            request.Length += request.StepTokens;
            if (request.StepLogitsRow < 0)
            {
                continue;
            }

            var token = GreedySampling.ArgMax(_decoder.GetLogits(request.StepLogitsRow));
            if (request.Accept(token, _endOfSequenceToken))
            {
                _ = _active.Remove(request);
            }
        }

        StepCompleted?.Invoke(new BatchStepTrace(entries));
    }

    private void RetireCancelled()
    {
        foreach (var request in _active.Concat(_waiting).Where(request => request.Cancellation.IsCancellationRequested).ToArray())
        {
            request.Cancel();
            _ = _active.Remove(request);
            _ = _waiting.Remove(request);
        }
    }

    private void FailActive(Exception exception)
    {
        foreach (var request in _active)
        {
            request.Fail(exception);
        }

        _active.Clear();
    }
}
