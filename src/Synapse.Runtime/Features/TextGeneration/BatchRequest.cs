using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using ManagedCode.Synapse.Runtime.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>One scheduled generation. Only the scheduler thread mutates it after admission.</summary>
internal sealed class BatchRequest(int[] prompt, int maximumNewTokens, CancellationToken cancellation)
{
    private readonly Stopwatch _timer = Stopwatch.StartNew();
    private TimeSpan _timeToFirstToken;

    public int[] Prompt { get; } = prompt;

    public int MaximumNewTokens { get; } = maximumNewTokens;

    public CancellationToken Cancellation { get; } = cancellation;

    public TaskCompletionSource<TextGenerationResult> Completion { get; } =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    public CancellationTokenRegistration Registration { get; set; }

    public List<int> Generated { get; } = [];

    /// <summary>Tokens already written to this request's KV slot.</summary>
    public int Length { get; set; }

    public int Slot { get; set; }

    public int StepPrefill { get; private set; }

    public int StepDecode { get; private set; }

    public int StepLogitsRow { get; private set; } = -1;

    public int StepTokens => StepPrefill + StepDecode;

    public bool InPrefill => Length < Prompt.Length;

    public void BeginStep(int prefill, int decode, int logitsRow)
    {
        StepPrefill = prefill;
        StepDecode = decode;
        StepLogitsRow = logitsRow;
    }

    /// <summary>Records a sampled token and returns <see langword="true"/> when the request is complete.</summary>
    public bool Accept(int token, int endOfSequenceToken)
    {
        Generated.Add(token);
        if (Generated.Count == 1)
        {
            _timeToFirstToken = _timer.Elapsed;
        }

        if (token != endOfSequenceToken && Generated.Count < MaximumNewTokens)
        {
            return false;
        }

        Registration.Dispose();
        _ = Completion.TrySetResult(new TextGenerationResult(Prompt, [.. Generated], _timeToFirstToken, _timer.Elapsed));
        return true;
    }

    public void Cancel()
    {
        Registration.Dispose();
        _ = Completion.TrySetCanceled(Cancellation);
    }

    public void Fail(Exception exception)
    {
        Registration.Dispose();
        _ = Completion.TrySetException(exception);
    }
}

/// <summary>Greedy token selection shared by every backend.</summary>
internal static class GreedySampling
{
    /// <summary>First index of the maximum logit, identical to the scalar scan for finite values.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.Length < Vector128<float>.Count * 2)
        {
            return Q8Operators.ArgMax(values);
        }

        var maximum = Vector128.Create(float.NegativeInfinity);
        var index = 0;
        for (; index <= values.Length - Vector128<float>.Count; index += Vector128<float>.Count)
        {
            maximum = Vector128.Max(maximum, Vector128.Create(values.Slice(index, Vector128<float>.Count)));
        }

        var best = MathF.Max(
            MathF.Max(maximum.GetElement(0), maximum.GetElement(1)),
            MathF.Max(maximum.GetElement(2), maximum.GetElement(3)));
        for (; index < values.Length; index++)
        {
            best = MathF.Max(best, values[index]);
        }

        var found = values.IndexOf(best);
        return found >= 0 ? found : Q8Operators.ArgMax(values);
    }
}
