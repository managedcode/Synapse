using ManagedCode.Synapse.Runtime.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Permanent oracle path: FP32 activations times dequantized Q8_0 weights in scalar loops, one prompt token
/// at a time (ADR-006 <c>reference</c>).
/// </summary>
internal sealed class Qwen2ReferenceExecutor : IQwen2Executor
{
    private readonly GgufFile _file;
    private readonly Qwen2Weights _weights;
    private readonly Qwen2Dimensions _dimensions;
    private readonly Qwen2Scratch _scratch;
    private readonly Qwen2KvCache _cache;
    private readonly ParallelOptions _parallelOptions;

    public Qwen2ReferenceExecutor(GgufFile file, Qwen2Weights weights, Qwen2Dimensions dimensions, int threads)
    {
        _file = file;
        _weights = weights;
        _dimensions = dimensions;
        _scratch = new Qwen2Scratch(
            dimensions.HiddenSize,
            dimensions.KvWidth,
            dimensions.FeedForwardSize,
            dimensions.VocabularySize,
            dimensions.ContextSize);
        _cache = new Qwen2KvCache(dimensions.LayerCount, dimensions.ContextSize, dimensions.KvWidth);
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = threads };
    }

    public string RuntimeProfile => "reference-qwen2-q8_0";

    public string KernelImplementation => "reference-scalar-fp32";

    public ReadOnlyMemory<float> Prefill(IReadOnlyList<int> tokens)
    {
        for (var index = 0; index < tokens.Count - 1; index++)
        {
            _ = Forward(tokens[index], index, computeLogits: false);
        }

        return Forward(tokens[^1], tokens.Count - 1, computeLogits: true);
    }

    public ReadOnlyMemory<float> Decode(int token, int position) => Forward(token, position, computeLogits: true);

    public void Dispose()
    {
    }

    private float[] Forward(int token, int position, bool computeLogits)
    {
        Q8Operators.ReadRow(_file, _weights.TokenEmbedding, token, _scratch.Hidden);
        for (var layer = 0; layer < _dimensions.LayerCount; layer++)
        {
            ExecuteLayer(layer, position, _weights.Layers[layer]);
        }

        if (!computeLogits)
        {
            return _scratch.Logits;
        }

        Q8Operators.RmsNorm(_scratch.Hidden, _weights.OutputNorm, _dimensions.RmsNormEpsilon, _scratch.Normalized);
        Q8Operators.Multiply(_file, _weights.Output, _scratch.Normalized, _scratch.Logits, _parallelOptions);
        return _scratch.Logits;
    }

    private void ExecuteLayer(int layer, int position, Qwen2LayerWeights weights)
    {
        var epsilon = _dimensions.RmsNormEpsilon;
        _scratch.Hidden.CopyTo(_scratch.Residual, 0);
        Q8Operators.RmsNorm(_scratch.Hidden, weights.AttentionNorm, epsilon, _scratch.Normalized);
        Q8Operators.Multiply(_file, weights.Query, _scratch.Normalized, _scratch.Query, _parallelOptions);
        Q8Operators.Multiply(_file, weights.Key, _scratch.Normalized, _scratch.Key, _parallelOptions);
        Q8Operators.Multiply(_file, weights.Value, _scratch.Normalized, _scratch.Value, _parallelOptions);
        Q8Operators.AddBiasInPlace(_scratch.Query, weights.QueryBias);
        Q8Operators.AddBiasInPlace(_scratch.Key, weights.KeyBias);
        Q8Operators.AddBiasInPlace(_scratch.Value, weights.ValueBias);
        var headDimension = _dimensions.HeadDimension;
        Qwen2Attention.ApplyRope(_scratch.Query, _dimensions.AttentionHeads, headDimension, position, _dimensions.RopeTheta);
        Qwen2Attention.ApplyRope(_scratch.Key, _dimensions.KeyValueHeads, headDimension, position, _dimensions.RopeTheta);
        _cache.Store(layer, position, _scratch.Key, _scratch.Value);
        Qwen2Attention.Execute(
            _cache,
            _scratch,
            layer,
            position,
            _dimensions.AttentionHeads,
            _dimensions.KeyValueHeads,
            headDimension);
        Q8Operators.Multiply(_file, weights.AttentionOutput, _scratch.Attention, _scratch.Projection, _parallelOptions);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);

        _scratch.Hidden.CopyTo(_scratch.Residual, 0);
        Q8Operators.RmsNorm(_scratch.Hidden, weights.FeedForwardNorm, epsilon, _scratch.Normalized);
        Q8Operators.Multiply(_file, weights.FeedForwardGate, _scratch.Normalized, _scratch.Gate, _parallelOptions);
        Q8Operators.Multiply(_file, weights.FeedForwardUp, _scratch.Normalized, _scratch.Up, _parallelOptions);
        Q8Operators.SwiGluInPlace(_scratch.Gate, _scratch.Up);
        Q8Operators.Multiply(_file, weights.FeedForwardDown, _scratch.Gate, _scratch.Projection, _parallelOptions);
        _scratch.Residual.CopyTo(_scratch.Hidden, 0);
        Q8Operators.AddInPlace(_scratch.Hidden, _scratch.Projection);
    }
}
