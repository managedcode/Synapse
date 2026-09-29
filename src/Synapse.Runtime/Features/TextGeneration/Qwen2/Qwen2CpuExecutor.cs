using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Optimized Qwen2 path (ADR-006 <c>managed</c> and <c>native</c>): Q8_0 activations, four fused parallel
/// regions per layer, and batched steps whose tokens may belong to different KV slots (ADR-007). Every token
/// runs the same per-token math whatever else shares its step.
/// </summary>
internal sealed class Qwen2CpuExecutor : IDecoderExecutor, IBatchDecoder
{
    private readonly GgufFile _file;
    private readonly Qwen2Weights _weights;
    private readonly DecoderDimensions _dimensions;
    private readonly Qwen2CpuLayer[] _layers;
    private readonly Q8Matrix _output;
    private readonly CpuWorkerPool _pool;
    private readonly Qwen2CpuScratch _scratch;
    private readonly Qwen2KvSlots _slots;
    private readonly BatchToken[] _batch;
    private readonly Qwen2RopeTable _rope;
    private readonly Qwen2CpuAttention _attention;
    private readonly Qwen2QkvWork _qkv;
    private readonly Qwen2ResidualWork _residual;
    private readonly Qwen2GateUpWork _gateUp;
    private readonly Qwen2LogitsWork _logits;
    private readonly int _prefillTokens;

    public Qwen2CpuExecutor(
        GgufFile file,
        Qwen2Weights weights,
        DecoderDimensions dimensions,
        Q8MatrixKernel kernel,
        string runtimeProfile,
        CpuWorkerPool pool,
        DecoderStepCapacity capacity)
    {
        _file = file;
        _weights = weights;
        _dimensions = dimensions;
        RuntimeProfile = runtimeProfile;
        KernelImplementation = kernel.Name;
        _layers = [.. weights.Layers.Select(layer => Qwen2CpuLayer.Create(file, layer))];
        _output = Q8Matrix.FromTensor(file, weights.Output);
        _pool = pool;
        _scratch = new Qwen2CpuScratch(dimensions, capacity.StepTokens, capacity.LogitsRows);
        _slots = new Qwen2KvSlots(dimensions, capacity.SessionSlots);
        _batch = new BatchToken[capacity.StepTokens];
        _prefillTokens = capacity.PrefillTokens;
        _rope = new Qwen2RopeTable(
            new RopeFrequencies(dimensions.HeadDimension, dimensions.RopeTheta, dimensions.RopeScaling),
            dimensions.ContextSize);
        _attention = new Qwen2CpuAttention(dimensions, _slots, _batch, _scratch, pool);
        _qkv = new Qwen2QkvWork(kernel, _scratch);
        _residual = new Qwen2ResidualWork(kernel, _scratch.Hidden, pool.ThreadCount);
        _gateUp = new Qwen2GateUpWork(kernel, _scratch, pool.ThreadCount);
        _logits = new Qwen2LogitsWork(kernel, _scratch);
    }

    public string RuntimeProfile { get; }

    public string KernelImplementation { get; }

    public int StepTokenCapacity => _batch.Length;

    public int SessionSlots => _slots.Count;

    public int VocabularySize => _dimensions.VocabularySize;

    public int LogitsRowCapacity => _scratch.LogitsRows;

    public long AllocatedKvBytes => _slots.AllocatedBytes;

    public void Reserve(int positions) => _slots[0].Reserve(positions);

    public ReadOnlyMemory<float> PrefillFrom(IReadOnlyList<int> tokens, int first, Action<int>? evaluated)
    {
        for (var start = first; start < tokens.Count; start += _prefillTokens)
        {
            var count = Math.Min(_prefillTokens, tokens.Count - start);
            for (var index = 0; index < count; index++)
            {
                var position = start + index;
                _batch[index] = new BatchToken(0, tokens[position], position, position == tokens.Count - 1 ? 0 : -1);
            }

            Forward(_batch.AsSpan(0, count), promptStart: 0);
            evaluated?.Invoke(start + count);
        }

        return LogitsRow(0);
    }

    public ReadOnlyMemory<float> Decode(int token, int position)
    {
        _batch[0] = new BatchToken(0, token, position, 0);
        Forward(_batch.AsSpan(0, 1), promptStart: 1);
        return LogitsRow(0);
    }

    public void Forward(ReadOnlySpan<BatchToken> tokens, int promptStart)
    {
        var logitsRows = Validate(tokens);
        tokens.CopyTo(_batch);
        var hiddenSize = _dimensions.HiddenSize;
        for (var index = 0; index < tokens.Length; index++)
        {
            Q8Operators.ReadRow(
                _file,
                _weights.TokenEmbedding,
                tokens[index].Token,
                _scratch.Hidden.AsSpan(index * hiddenSize, hiddenSize));
        }

        for (var layer = 0; layer < _dimensions.LayerCount; layer++)
        {
            ExecuteLayer(layer, tokens.Length, promptStart);
        }

        if (logitsRows > 0)
        {
            ComputeLogits(tokens, logitsRows);
        }
    }

    public ReadOnlySpan<float> GetLogits(int row) => LogitsRow(row).Span;

    public void Dispose() => _pool.Dispose();

    private ReadOnlyMemory<float> LogitsRow(int row) =>
        new(_scratch.Logits, row * _dimensions.VocabularySize, _dimensions.VocabularySize);

    private int Validate(ReadOnlySpan<BatchToken> tokens)
    {
        if (tokens.IsEmpty || tokens.Length > _batch.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), $"A step holds 1..{_batch.Length} tokens.");
        }

        var rows = 0;
        foreach (var token in tokens)
        {
            if ((uint)token.Slot >= (uint)_slots.Count ||
                (uint)token.Token >= (uint)_dimensions.VocabularySize ||
                (uint)token.Position >= (uint)_dimensions.ContextSize ||
                token.LogitsRow >= _scratch.LogitsRows)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Invalid batch token {token}.");
            }

            rows += token.LogitsRow >= 0 ? 1 : 0;
        }

        return rows;
    }

    private void ExecuteLayer(int layer, int count, int promptStart)
    {
        var weights = _weights.Layers[layer];
        var matrices = _layers[layer];
        var threads = _pool.ThreadCount;
        NormalizeAndQuantize(weights.AttentionNorm, count);
        _qkv.Prepare(matrices, weights, count, threads);
        _pool.Run(_qkv);
        RotateAndStore(layer, count);
        _attention.Execute(layer, count, promptStart);
        Quantize(_scratch.Attention, _scratch.HiddenActivations, count);
        _residual.Prepare(matrices.Output, _scratch.HiddenActivations, count, threads);
        _pool.Run(_residual);

        NormalizeAndQuantize(weights.FeedForwardNorm, count);
        _gateUp.Prepare(matrices, count, threads);
        _pool.Run(_gateUp);
        Quantize(_scratch.FeedForward, _scratch.FeedForwardActivations, count);
        _residual.Prepare(matrices.Down, _scratch.FeedForwardActivations, count, threads);
        _pool.Run(_residual);
    }

    private void RotateAndStore(int layer, int count)
    {
        var hiddenSize = _dimensions.HiddenSize;
        var kvWidth = _dimensions.KvWidth;
        for (var index = 0; index < count; index++)
        {
            var (slot, _, position, _) = _batch[index];
            _rope.Apply(_scratch.Query.AsSpan(index * hiddenSize, hiddenSize), _dimensions.AttentionHeads, position);
            var key = _scratch.Key.AsSpan(index * kvWidth, kvWidth);
            _rope.Apply(key, _dimensions.KeyValueHeads, position);
            _slots[slot].Store(layer, position, key, _scratch.Value.AsSpan(index * kvWidth, kvWidth));
        }
    }

    /// <summary>Normalizes each logits token into activation row <c>LogitsRow</c>; one vocabulary pass serves all rows.</summary>
    private void ComputeLogits(ReadOnlySpan<BatchToken> tokens, int rows)
    {
        var hiddenSize = _dimensions.HiddenSize;
        for (var index = 0; index < tokens.Length; index++)
        {
            if (tokens[index].LogitsRow < 0)
            {
                continue;
            }

            Q8Operators.RmsNorm(
                _scratch.Hidden.AsSpan(index * hiddenSize, hiddenSize),
                _weights.OutputNorm,
                _dimensions.RmsNormEpsilon,
                _scratch.Normalized);
            _scratch.HiddenActivations.Quantize(tokens[index].LogitsRow, _scratch.Normalized);
        }

        _logits.Prepare(_output, rows, _pool.ThreadCount);
        _pool.Run(_logits);
    }

    private void NormalizeAndQuantize(float[] normWeights, int count)
    {
        var hiddenSize = _dimensions.HiddenSize;
        for (var index = 0; index < count; index++)
        {
            Q8Operators.RmsNorm(
                _scratch.Hidden.AsSpan(index * hiddenSize, hiddenSize),
                normWeights,
                _dimensions.RmsNormEpsilon,
                _scratch.Normalized);
            _scratch.HiddenActivations.Quantize(index, _scratch.Normalized);
        }
    }

    private static void Quantize(float[] rows, Q8ActivationBuffer activations, int count)
    {
        for (var index = 0; index < count; index++)
        {
            activations.Quantize(index, rows.AsSpan(index * activations.Columns, activations.Columns));
        }
    }
}
