using System.Diagnostics;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Executes the Qwen2 dense decoder directly from Q8_0 GGUF weights using managed C# operators.
/// </summary>
public sealed class Qwen2Model : ITextGenerationModel
{
    private const int EndOfSequenceToken = 151645;
    private readonly GgufFile _file;
    private readonly Qwen2Weights _weights;
    private readonly Qwen2Scratch _scratch;
    private readonly Qwen2KvCache _cache;
    private readonly ParallelOptions _parallelOptions;
    private readonly int _headDimension;
    private readonly int _kvWidth;

    private Qwen2Model(GgufFile file, int contextSize, int maximumParallelism)
    {
        _file = file;
        LayerCount = file.GetRequiredInt32("qwen2.block_count");
        HiddenSize = file.GetRequiredInt32("qwen2.embedding_length");
        FeedForwardSize = file.GetRequiredInt32("qwen2.feed_forward_length");
        AttentionHeads = file.GetRequiredInt32("qwen2.attention.head_count");
        KeyValueHeads = file.GetRequiredInt32("qwen2.attention.head_count_kv");
        RopeTheta = file.GetRequiredSingle("qwen2.rope.freq_base");
        RmsNormEpsilon = file.GetRequiredSingle("qwen2.attention.layer_norm_rms_epsilon");
        _headDimension = HiddenSize / AttentionHeads;
        _kvWidth = KeyValueHeads * _headDimension;
        var maximumContextSize = file.GetRequiredInt32("qwen2.context_length");
        ContextSize = Math.Min(contextSize, maximumContextSize);
        Graph = Qwen2GraphBuilder.Build(
            file,
            LayerCount,
            HiddenSize,
            FeedForwardSize,
            _kvWidth,
            maximumContextSize);
        var verification = ModelGraphVerifier.Verify(Graph);
        if (!verification.IsValid)
        {
            throw new InvalidDataException(
                "Qwen2 model graph verification failed: " +
                string.Join(" | ", verification.Diagnostics.Select(diagnostic => diagnostic.Message)));
        }

        _weights = Qwen2WeightLoader.Load(file, LayerCount);
        VocabularySize = checked((int)_weights.TokenEmbedding.Dimensions[1]);
        _scratch = new Qwen2Scratch(HiddenSize, _kvWidth, FeedForwardSize, VocabularySize, ContextSize);
        _cache = new Qwen2KvCache(LayerCount, ContextSize, _kvWidth);
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maximumParallelism };
    }

    /// <summary>Number of transformer blocks.</summary>
    public int LayerCount { get; }

    /// <summary>Model hidden width.</summary>
    public int HiddenSize { get; }

    /// <summary>Feed-forward intermediate width.</summary>
    public int FeedForwardSize { get; }

    /// <summary>Number of query attention heads.</summary>
    public int AttentionHeads { get; }

    /// <summary>Number of grouped key/value heads.</summary>
    public int KeyValueHeads { get; }

    /// <summary>Maximum tokens accepted by this model instance.</summary>
    public int ContextSize { get; }

    /// <summary>Vocabulary row count from the GGUF embedding tensor.</summary>
    public int VocabularySize { get; }

    /// <summary>Rotary embedding frequency base.</summary>
    public float RopeTheta { get; }

    /// <summary>RMS normalization epsilon.</summary>
    public float RmsNormEpsilon { get; }

    /// <inheritdoc />
    public string Architecture => "qwen2";

    /// <inheritdoc />
    public string RuntimeProfile => "managed-qwen2-q8_0";

    /// <summary>Verified portable dense graph corresponding to this loaded model.</summary>
    public ModelGraph Graph { get; }

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF file through a read-only memory map.</summary>
    public static Qwen2Model Load(string modelPath, int contextSize = 512) =>
        Load(modelPath, contextSize, Environment.ProcessorCount);

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF with an explicit CPU parallelism limit.</summary>
    public static Qwen2Model Load(string modelPath, int contextSize, int maximumParallelism)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumParallelism);

        var file = GgufFile.Open(modelPath);
        try
        {
            return new Qwen2Model(file, contextSize, maximumParallelism);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    internal static Qwen2Model Load(GgufFile file, int contextSize, int maximumParallelism) =>
        new(file, contextSize, maximumParallelism);

    /// <summary>Runs greedy generation from already-tokenized input IDs.</summary>
    public TextGenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);
        if (promptTokens.Count == 0 || maximumNewTokens <= 0 ||
            promptTokens.Count + maximumNewTokens > ContextSize)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumNewTokens));
        }

        var timer = Stopwatch.StartNew();
        for (var index = 0; index < promptTokens.Count - 1; index++)
        {
            _ = Forward(promptTokens[index], index, computeLogits: false);
        }

        var logits = Forward(promptTokens[promptTokens.Count - 1], promptTokens.Count - 1, computeLogits: true);
        var generated = new List<int>(maximumNewTokens);
        var timeToFirstToken = TimeSpan.Zero;
        for (var index = 0; index < maximumNewTokens; index++)
        {
            var token = Q8Operators.ArgMax(logits);
            generated.Add(token);
            if (index == 0)
            {
                timeToFirstToken = timer.Elapsed;
            }

            if (token == EndOfSequenceToken || index + 1 == maximumNewTokens)
            {
                break;
            }

            logits = Forward(token, promptTokens.Count + index, computeLogits: true);
        }

        timer.Stop();
        return new TextGenerationResult([.. promptTokens], generated, timeToFirstToken, timer.Elapsed);
    }

    /// <inheritdoc />
    public void Dispose() => _file.Dispose();

    private float[] Forward(int token, int position, bool computeLogits)
    {
        if ((uint)token >= (uint)VocabularySize || (uint)position >= (uint)ContextSize)
        {
            throw new ArgumentOutOfRangeException(nameof(token));
        }

        Q8Operators.ReadRow(_file, _weights.TokenEmbedding, token, _scratch.Hidden);
        for (var layer = 0; layer < LayerCount; layer++)
        {
            ExecuteLayer(layer, position, _weights.Layers[layer]);
        }

        if (!computeLogits)
        {
            return _scratch.Logits;
        }

        Q8Operators.RmsNorm(
            _scratch.Hidden,
            _weights.OutputNorm,
            RmsNormEpsilon,
            _scratch.Normalized);
        Q8Operators.Multiply(_file, _weights.Output, _scratch.Normalized, _scratch.Logits, _parallelOptions);
        return _scratch.Logits;
    }

    private void ExecuteLayer(int layer, int position, Qwen2LayerWeights weights)
    {
        _scratch.Hidden.CopyTo(_scratch.Residual, 0);
        Q8Operators.RmsNorm(
            _scratch.Hidden,
            weights.AttentionNorm,
            RmsNormEpsilon,
            _scratch.Normalized);
        Q8Operators.Multiply(_file, weights.Query, _scratch.Normalized, _scratch.Query, _parallelOptions);
        Q8Operators.Multiply(_file, weights.Key, _scratch.Normalized, _scratch.Key, _parallelOptions);
        Q8Operators.Multiply(_file, weights.Value, _scratch.Normalized, _scratch.Value, _parallelOptions);
        Q8Operators.AddBiasInPlace(_scratch.Query, weights.QueryBias);
        Q8Operators.AddBiasInPlace(_scratch.Key, weights.KeyBias);
        Q8Operators.AddBiasInPlace(_scratch.Value, weights.ValueBias);
        Qwen2Attention.ApplyRope(_scratch.Query, AttentionHeads, _headDimension, position, RopeTheta);
        Qwen2Attention.ApplyRope(_scratch.Key, KeyValueHeads, _headDimension, position, RopeTheta);
        _cache.Store(layer, position, _scratch.Key, _scratch.Value);
        Qwen2Attention.Execute(
            _cache,
            _scratch,
            layer,
            position,
            AttentionHeads,
            KeyValueHeads,
            _headDimension);
        Q8Operators.Multiply(
            _file,
            weights.AttentionOutput,
            _scratch.Attention,
            _scratch.Projection,
            _parallelOptions);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);

        _scratch.Hidden.CopyTo(_scratch.Residual, 0);
        Q8Operators.RmsNorm(
            _scratch.Hidden,
            weights.FeedForwardNorm,
            RmsNormEpsilon,
            _scratch.Normalized);
        Q8Operators.Multiply(_file, weights.FeedForwardGate, _scratch.Normalized, _scratch.Gate, _parallelOptions);
        Q8Operators.Multiply(_file, weights.FeedForwardUp, _scratch.Normalized, _scratch.Up, _parallelOptions);
        Q8Operators.SwiGluInPlace(_scratch.Gate, _scratch.Up);
        Q8Operators.Multiply(_file, weights.FeedForwardDown, _scratch.Gate, _scratch.Projection, _parallelOptions);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);
    }

}
