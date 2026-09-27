using System.Diagnostics;
using ManagedCode.Synapse.Runtime.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Executes the Qwen2 dense decoder directly from Q8_0 GGUF weights using managed C# operators.
/// </summary>
public sealed class Qwen2Model : IDisposable
{
    private const int EndOfSequenceToken = 151645;
    private readonly GgufFile _file;
    private readonly Qwen2Weights _weights;
    private readonly Qwen2Scratch _scratch;
    private readonly Qwen2KvCache _cache;
    private readonly int _headDimension;
    private readonly int _kvWidth;

    private Qwen2Model(GgufFile file, int contextSize)
    {
        _file = file;
        if (!string.Equals(file.GetRequiredString("general.architecture"), "qwen2", StringComparison.Ordinal))
        {
            throw new NotSupportedException("The initial managed engine supports only Qwen2 GGUF models.");
        }

        LayerCount = file.GetRequiredInt32("qwen2.block_count");
        HiddenSize = file.GetRequiredInt32("qwen2.embedding_length");
        FeedForwardSize = file.GetRequiredInt32("qwen2.feed_forward_length");
        AttentionHeads = file.GetRequiredInt32("qwen2.attention.head_count");
        KeyValueHeads = file.GetRequiredInt32("qwen2.attention.head_count_kv");
        RopeTheta = file.GetRequiredSingle("qwen2.rope.freq_base");
        RmsNormEpsilon = file.GetRequiredSingle("qwen2.attention.layer_norm_rms_epsilon");
        _headDimension = HiddenSize / AttentionHeads;
        _kvWidth = KeyValueHeads * _headDimension;
        ContextSize = Math.Min(contextSize, file.GetRequiredInt32("qwen2.context_length"));
        _weights = LoadWeights(file, LayerCount);
        VocabularySize = checked((int)_weights.TokenEmbedding.Dimensions[1]);
        _scratch = new Qwen2Scratch(HiddenSize, _kvWidth, FeedForwardSize, VocabularySize, ContextSize);
        _cache = new Qwen2KvCache(LayerCount, ContextSize, _kvWidth);
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

    /// <summary>Loads a supported Qwen2 Q8_0 GGUF file through a read-only memory map.</summary>
    public static Qwen2Model Load(string modelPath, int contextSize = 512)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSize);

        var file = GgufFile.Open(modelPath);
        try
        {
            return new Qwen2Model(file, contextSize);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>Runs greedy generation from already-tokenized input IDs.</summary>
    public Qwen2GenerationResult Generate(IReadOnlyList<int> promptTokens, int maximumNewTokens)
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
        for (var index = 0; index < maximumNewTokens; index++)
        {
            var token = Q8Operators.ArgMax(logits);
            generated.Add(token);
            if (token == EndOfSequenceToken || index + 1 == maximumNewTokens)
            {
                break;
            }

            logits = Forward(token, promptTokens.Count + index, computeLogits: true);
        }

        timer.Stop();
        return new Qwen2GenerationResult([.. promptTokens], generated, timer.Elapsed);
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
        Q8Operators.Multiply(_file, _weights.Output, _scratch.Normalized, _scratch.Logits);
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
        Q8Operators.Multiply(_file, weights.Query, _scratch.Normalized, _scratch.Query);
        Q8Operators.Multiply(_file, weights.Key, _scratch.Normalized, _scratch.Key);
        Q8Operators.Multiply(_file, weights.Value, _scratch.Normalized, _scratch.Value);
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
            _scratch.Projection);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);

        _scratch.Hidden.CopyTo(_scratch.Residual, 0);
        Q8Operators.RmsNorm(
            _scratch.Hidden,
            weights.FeedForwardNorm,
            RmsNormEpsilon,
            _scratch.Normalized);
        Q8Operators.Multiply(_file, weights.FeedForwardGate, _scratch.Normalized, _scratch.Gate);
        Q8Operators.Multiply(_file, weights.FeedForwardUp, _scratch.Normalized, _scratch.Up);
        Q8Operators.SwiGluInPlace(_scratch.Gate, _scratch.Up);
        Q8Operators.Multiply(_file, weights.FeedForwardDown, _scratch.Gate, _scratch.Projection);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);
    }

    private static Qwen2Weights LoadWeights(GgufFile file, int layerCount)
    {
        var layers = new Qwen2LayerWeights[layerCount];
        for (var layer = 0; layer < layerCount; layer++)
        {
            var prefix = $"blk.{layer}";
            layers[layer] = new Qwen2LayerWeights(
                file.ReadFloat32Tensor($"{prefix}.attn_norm.weight"),
                file.GetRequiredTensor($"{prefix}.attn_q.weight"),
                file.GetRequiredTensor($"{prefix}.attn_k.weight"),
                file.GetRequiredTensor($"{prefix}.attn_v.weight"),
                file.ReadFloat32Tensor($"{prefix}.attn_q.bias"),
                file.ReadFloat32Tensor($"{prefix}.attn_k.bias"),
                file.ReadFloat32Tensor($"{prefix}.attn_v.bias"),
                file.GetRequiredTensor($"{prefix}.attn_output.weight"),
                file.ReadFloat32Tensor($"{prefix}.ffn_norm.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_gate.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_up.weight"),
                file.GetRequiredTensor($"{prefix}.ffn_down.weight"));
        }

        return new Qwen2Weights(
            file.GetRequiredTensor("token_embd.weight"),
            layers,
            file.ReadFloat32Tensor("output_norm.weight"),
            file.GetRequiredTensor("output.weight"));
    }
}
