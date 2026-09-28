using System.Diagnostics;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Executes the Qwen2 dense decoder directly from memory-mapped Q8_0 GGUF weights on an explicit CPU kernel
/// backend (ADR-006). Concurrent <see cref="GenerateAsync"/> calls share batched steps (ADR-007).
/// </summary>
public sealed class Qwen2Model : ITextGenerationModel
{
    private const int EndOfSequenceToken = 151645;
    private readonly GgufFile _file;
    private readonly IQwen2Executor _executor;
    private readonly ContinuousBatchScheduler? _scheduler;
    private readonly object _executionGate = new();

    private Qwen2Model(GgufFile file, ModelLoadOptions options)
    {
        _file = file;
        Dimensions = Qwen2ModelComposition.ReadDimensions(file, options);
        var pool = options.KernelBackend == CpuKernelBackend.Reference ? null : new CpuWorkerPool(options.MaximumParallelism);
        try
        {
            Qwen2ModelComposition.PrefetchWeights(file, pool);
            Graph = Qwen2ModelComposition.BuildVerifiedGraph(file, Dimensions);
            var weights = Qwen2WeightLoader.Load(file, Dimensions.LayerCount);
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
    public string RuntimeProfile => _executor.RuntimeProfile;

    /// <inheritdoc />
    public string KernelImplementation => _executor.KernelImplementation;

    /// <summary>Verified portable dense graph corresponding to this loaded model.</summary>
    public ModelGraph Graph { get; }

    internal Qwen2Dimensions Dimensions { get; }

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
    public TextGenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens)
    {
        ValidatePrompt(promptTokens, maximumNewTokens, allowEmptyOutput: false);
        lock (_executionGate)
        {
            var timer = Stopwatch.StartNew();
            var logits = _executor.Prefill(promptTokens);
            var generated = new List<int>(maximumNewTokens);
            var timeToFirstToken = TimeSpan.Zero;
            for (var index = 0; index < maximumNewTokens; index++)
            {
                var token = GreedySampling.ArgMax(logits.Span);
                generated.Add(token);
                timeToFirstToken = index == 0 ? timer.Elapsed : timeToFirstToken;
                if (token == EndOfSequenceToken || index + 1 == maximumNewTokens)
                {
                    break;
                }

                logits = _executor.Decode(token, promptTokens.Count + index);
            }

            return new TextGenerationResult([.. promptTokens], generated, timeToFirstToken, timer.Elapsed);
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
        lock (_executionGate)
        {
            return _executor.Prefill(promptTokens).ToArray();
        }
    }

    /// <summary>Prefills <paramref name="promptTokens"/>, then decodes each continuation token through the KV cache.</summary>
    internal float[] EvaluateIncrementalLogits(IReadOnlyList<int> promptTokens, IReadOnlyList<int> continuation)
    {
        ValidatePrompt([.. promptTokens, .. continuation], 0, allowEmptyOutput: true);
        lock (_executionGate)
        {
            var logits = _executor.Prefill(promptTokens);
            for (var index = 0; index < continuation.Count; index++)
            {
                logits = _executor.Decode(continuation[index], promptTokens.Count + index);
            }

            return logits.ToArray();
        }
    }

    private ContinuousBatchScheduler RequireScheduler() => _scheduler
        ?? throw new NotSupportedException("The reference backend serializes requests and has no batching scheduler.");

    private void ValidatePrompt(IReadOnlyList<int> promptTokens, int maximumNewTokens, bool allowEmptyOutput)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count == 0 || maximumNewTokens < (allowEmptyOutput ? 0 : 1) ||
            promptTokens.Count + maximumNewTokens > ContextSize)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumNewTokens));
        }

        foreach (var token in promptTokens)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)token, (uint)VocabularySize, nameof(promptTokens));
        }
    }
}
