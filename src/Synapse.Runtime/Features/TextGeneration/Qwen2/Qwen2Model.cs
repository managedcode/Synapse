using System.Diagnostics;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.Speculation;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Executes the Qwen2 dense decoder directly from memory-mapped Q8_0 GGUF weights on an explicit CPU kernel
/// backend (ADR-006). Concurrent <see cref="GenerateAsync"/> calls share batched steps (ADR-007).
/// </summary>
public sealed class Qwen2Model : ITextGenerationModel
{
    internal const int EndOfSequenceToken = 151645;
    private readonly GgufFile _file;
    private readonly IDecoderExecutor _executor;
    private readonly ContinuousBatchScheduler? _scheduler;
    private readonly object _executionGate = new();
    private readonly DirectSessionPrefix _direct;

    private Qwen2Model(GgufFile file, ModelLoadOptions options)
    {
        _file = file;
        Dimensions = Qwen2ModelComposition.ReadDimensions(file, options);
        _direct = new DirectSessionPrefix(options.ReusePromptPrefix);
        var pool = options.KernelBackend == KernelBackend.Reference ? null : new CpuWorkerPool(options.MaximumParallelism);
        try
        {
            Qwen2ModelComposition.PrefetchWeights(file, Dimensions, pool);
            Graph = Qwen2ModelComposition.BuildVerifiedGraph(file, Dimensions);
            var weights = Qwen2WeightLoader.Load(file, Dimensions);
            _executor = Qwen2ModelComposition.CreateExecutor(file, weights, Dimensions, options, pool);
        }
        catch
        {
            pool?.Dispose();
            throw;
        }

        if (_executor is IBatchDecoder decoder)
        {
            _scheduler = new ContinuousBatchScheduler(decoder, _executionGate, EndOfSequenceToken);
        }
    }

    /// <summary>Published after every continuous-batching step; diagnostics and tests only.</summary>
    internal event Action<BatchStepTrace>? BatchStepCompleted
    {
        add => RequireScheduler().StepCompleted += value;
        remove => RequireScheduler().StepCompleted -= value;
    }

    /// <summary>Number of transformer blocks.</summary>
    public int LayerCount => Dimensions.LayerCount;

    /// <summary>Model hidden width.</summary>
    public int HiddenSize => Dimensions.HiddenSize;

    /// <summary>Feed-forward intermediate width.</summary>
    public int FeedForwardSize => Dimensions.FeedForwardSize;

    /// <summary>Number of query attention heads.</summary>
    public int AttentionHeads => Dimensions.AttentionHeads;

    /// <summary>Number of grouped key/value heads.</summary>
    public int KeyValueHeads => Dimensions.KeyValueHeads;

    /// <summary>Maximum tokens accepted by this model instance.</summary>
    public int ContextSize => Dimensions.ContextSize;

    /// <summary>Vocabulary row count from the GGUF embedding tensor.</summary>
    public int VocabularySize => Dimensions.VocabularySize;

    /// <summary>Rotary embedding frequency base.</summary>
    public float RopeTheta => Dimensions.RopeTheta;

    /// <summary>RMS normalization epsilon.</summary>
    public float RmsNormEpsilon => Dimensions.RmsNormEpsilon;

    /// <inheritdoc />
    public string Architecture => "qwen2";

    /// <inheritdoc />
    public string RuntimeProfile => _executor.RuntimeProfile +
        (Dimensions.RopeScaling is { } scaling ? "+" + scaling.Name : string.Empty) +
        (Dimensions.KvPages is { } pages ? "+" + pages.Name : string.Empty) +
        (Dimensions.LayerDrop is { } drop ? "+" + drop.Name : string.Empty);

    /// <inheritdoc />
    public string KernelImplementation => _executor.KernelImplementation;

    /// <summary>Verified portable dense graph corresponding to this loaded model.</summary>
    public ModelGraph Graph { get; }

    internal DecoderDimensions Dimensions { get; }

    /// <summary>Bytes of KV currently allocated across the executor's slots (ADR-017); waits for a running step.</summary>
    internal long AllocatedKvBytes
    {
        get
        {
            lock (_executionGate)
            {
                return _executor.AllocatedKvBytes;
            }
        }
    }

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF file through a read-only memory map.</summary>
    public static Qwen2Model Load(string modelPath, int contextSize = 512) =>
        Load(modelPath, contextSize, Environment.ProcessorCount);

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF with an explicit CPU parallelism limit.</summary>
    public static Qwen2Model Load(string modelPath, int contextSize, int maximumParallelism) =>
        Load(modelPath, new ModelLoadOptions { ContextSize = contextSize, MaximumParallelism = maximumParallelism });

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF with explicit limits and CPU kernel backend.</summary>
    public static Qwen2Model Load(string modelPath, ModelLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        var file = GgufFile.Open(modelPath);
        try
        {
            return new Qwen2Model(file, options);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    internal static Qwen2Model Load(GgufFile file, ModelLoadOptions options) => new(file, options);

    /// <summary>Runs greedy generation on the direct session; serialized with every other caller.</summary>
    public TextGenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens) =>
        Generate(promptTokens, maximumNewTokens, progress: null);

    /// <inheritdoc />
    public TextGenerationResult Generate(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        IProgress<GenerationProgress>? progress)
    {
        ValidatePrompt(promptTokens, maximumNewTokens, allowEmptyOutput: false);
        lock (_executionGate)
        {
            var timer = Stopwatch.StartNew();
            var promptCount = promptTokens.Count;
            var reused = _direct.Claim(promptTokens);
            _executor.Reserve(promptCount + maximumNewTokens);
            var logits = _executor.PrefillFrom(
                promptTokens,
                reused,
                progress is null ? null : evaluated => progress.Report(new(evaluated, promptCount, 0, timer.Elapsed)));
            var (generated, timeToFirstToken) = GreedyDecoding.Continue(
                _executor,
                logits,
                promptCount,
                maximumNewTokens,
                EndOfSequenceToken,
                timer,
                progress is null ? null : count => progress.Report(new(promptCount, promptCount, count, timer.Elapsed)));
            _direct.Remember(promptTokens, generated);
            return new TextGenerationResult([.. promptTokens], generated, timeToFirstToken, timer.Elapsed)
            {
                ReusedPromptTokens = reused,
            };
        }
    }

    /// <inheritdoc />
    public Task<TextGenerationResult> GenerateAsync(
        IReadOnlyList<int> promptTokens,
        int maximumNewTokens,
        CancellationToken cancellationToken)
    {
        ValidatePrompt(promptTokens, maximumNewTokens, allowEmptyOutput: false);
        return _scheduler is not null
            ? _scheduler.EnqueueAsync(promptTokens, maximumNewTokens, cancellationToken)
            : Task.Run(() => Generate(promptTokens, maximumNewTokens), cancellationToken);
    }

    /// <inheritdoc />
    public TokenScores Score(IReadOnlyList<int> tokens, int firstScoredPosition, IProgress<GenerationProgress>? progress)
    {
        ValidatePrompt(tokens, 0, allowEmptyOutput: true);
        ArgumentOutOfRangeException.ThrowIfNegative(firstScoredPosition);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(firstScoredPosition, tokens.Count - 2);
        return RunDirect(executor =>
        {
            executor.Reserve(tokens.Count);
            var timer = Stopwatch.StartNew();
            var evaluatedTotal = tokens.Count - 1;
            return TokenScoring.Score(
                executor,
                tokens,
                firstScoredPosition,
                progress is null ? null : evaluated => progress.Report(new(evaluated, evaluatedTotal, 0, timer.Elapsed)));
        });
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _scheduler?.Dispose();
        _executor.Dispose();
        _file.Dispose();
    }

    /// <summary>Evaluates a prompt from position zero and returns a copy of the final logits.</summary>
    internal float[] EvaluatePromptLogits(IReadOnlyList<int> promptTokens)
    {
        ValidatePrompt(promptTokens, 0, allowEmptyOutput: true);
        return RunDirect(executor =>
        {
            executor.Reserve(promptTokens.Count);
            return executor.Prefill(promptTokens).ToArray();
        });
    }

    /// <summary>Prefills <paramref name="promptTokens"/>, then decodes each continuation token through the KV cache.</summary>
    internal float[] EvaluateIncrementalLogits(IReadOnlyList<int> promptTokens, IReadOnlyList<int> continuation)
    {
        ValidatePrompt([.. promptTokens, .. continuation], 0, allowEmptyOutput: true);
        return RunDirect(executor =>
        {
            executor.Reserve(promptTokens.Count + continuation.Count);
            var logits = executor.Prefill(promptTokens);
            for (var index = 0; index < continuation.Count; index++)
            {
                logits = executor.Decode(continuation[index], promptTokens.Count + index);
            }

            return logits.ToArray();
        });
    }

    /// <summary>
    /// Runs <paramref name="work"/> on the direct slot under the execution gate, after clearing the prompt-prefix
    /// memory (ADR-018), which any other direct use invalidates.
    /// </summary>
    internal T RunDirect<T>(Func<IDecoderExecutor, T> work)
    {
        lock (_executionGate)
        {
            _direct.Clear();
            return work(_executor);
        }
    }

    /// <summary>Digest of the first <paramref name="count"/> vocabulary entries and every merge (ADR-020).</summary>
    internal string VocabularyDigest(int count) => TokenIdentity.Digest(_file, count);

    private ContinuousBatchScheduler RequireScheduler() => _scheduler
        ?? throw new NotSupportedException("The reference backend serializes requests and has no batching scheduler.");

    private void ValidatePrompt(IReadOnlyList<int> promptTokens, int maximumNewTokens, bool allowEmptyOutput) =>
        PromptLimits.Validate(promptTokens, maximumNewTokens, allowEmptyOutput, ContextSize, VocabularySize);
}
