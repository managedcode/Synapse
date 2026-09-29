using System.Runtime.CompilerServices;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.GpuKernels;

/// <summary>
/// Runs any <see cref="DenseDecoderLayout"/> on a GPU backend (ADR-012). The native library owns device buffers
/// and KV slots; this executor keeps the managed contract: batched steps across slots (ADR-007), explicit
/// validation, and C#-computed RoPE frequencies (ADR-013). It knows no model brand and no source format.
/// Numeric profile: Q8_0 weights times FP32 activations with FP32 accumulation.
/// </summary>
internal sealed unsafe class GpuDecoderExecutor : IDecoderExecutor, IBatchDecoder
{
    private const ulong NoTensor = ulong.MaxValue;
    private const uint EncodingQ8Zero = 8;
    private readonly NativeGpuLibrary _library;
    private readonly DecoderDimensions _dimensions;
    private readonly BatchToken[] _batch;
    private readonly float[] _logits;
    private readonly int _prefillTokens;
    private nint _model;

    public GpuDecoderExecutor(
        IMappedWeights weights,
        DenseDecoderLayout layout,
        KernelBackend backend,
        KvCachePrecision kvPrecision,
        DecoderStepCapacity capacity)
    {
        _library = NativeGpuLibrary.LoadFromApplicationDirectory();
        var device = _library.Probe(backend);
        _dimensions = layout.Dimensions;
        _batch = new BatchToken[capacity.StepTokens];
        _logits = GC.AllocateArray<float>(checked(capacity.LogitsRows * _dimensions.VocabularySize), pinned: true);
        _prefillTokens = capacity.PrefillTokens;
        SessionSlots = capacity.SessionSlots;
        var backendName = KernelBackendNames.ToName(backend);
        RuntimeProfile = $"{backendName}-{layout.Architecture}-q8_0xf32" +
            (kvPrecision == KvCachePrecision.Fp16 ? "-kvf16" : string.Empty);
        KernelImplementation = backendName +
            (device.AppleFamily > 0 ? $"-apple{device.AppleFamily}" : string.Empty) + "-" + Slug(device.Name);
        _model = Create(weights, layout, backend, kvPrecision, capacity);
    }

    public string RuntimeProfile { get; }

    public string KernelImplementation { get; }

    public int StepTokenCapacity => _batch.Length;

    public int SessionSlots { get; }

    public int VocabularySize => _dimensions.VocabularySize;

    public long AllocatedKvBytes => _model == 0 ? 0 : _library.KvBytes(_model);

    public void Reserve(int positions) => _library.Reserve(_model, 0, positions);

    public int LogitsRowCapacity => _logits.Length / _dimensions.VocabularySize;

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

    /// <remarks>KV page activation is a CPU profile (ADR-016), so the prompt boundary changes nothing here.</remarks>
    public void Forward(ReadOnlySpan<BatchToken> tokens, int promptStart)
    {
        ObjectDisposedException.ThrowIf(_model == 0, this);
        _ = Validate(tokens);
        // The whole pinned buffer is passed even for zero logits rows, so the native side never sees null.
        _library.Forward(_model, tokens, _logits);
    }

    public ReadOnlySpan<float> GetLogits(int row) => LogitsRow(row).Span;

    public void Dispose()
    {
        if (_model != 0)
        {
            _library.Destroy(_model);
            _model = 0;
        }
    }

    private ReadOnlyMemory<float> LogitsRow(int row) =>
        new(_logits, row * _dimensions.VocabularySize, _dimensions.VocabularySize);

    private int Validate(ReadOnlySpan<BatchToken> tokens)
    {
        if (tokens.IsEmpty || tokens.Length > _batch.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), $"A step holds 1..{_batch.Length} tokens.");
        }

        var rows = 0;
        foreach (var token in tokens)
        {
            if ((uint)token.Slot >= (uint)SessionSlots ||
                (uint)token.Token >= (uint)_dimensions.VocabularySize ||
                (uint)token.Position >= (uint)_dimensions.ContextSize ||
                (token.LogitsRow >= 0 && token.LogitsRow != rows))
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Invalid batch token {token}.");
            }

            rows += token.LogitsRow >= 0 ? 1 : 0;
        }

        return rows;
    }

    private nint Create(
        IMappedWeights weights,
        DenseDecoderLayout layout,
        KernelBackend backend,
        KvCachePrecision kvPrecision,
        DecoderStepCapacity capacity)
    {
        var dimensions = layout.Dimensions;
        var (cosines, sines) = BuildRopeTables(dimensions);
        var layers = layout.Layers.Select(layer => Offsets(weights, layer)).ToArray();
        fixed (NativeDecoderLayerOffsets* layerPointer = layers)
        fixed (float* cosinePointer = cosines)
        fixed (float* sinePointer = sines)
        {
            var description = new NativeDecoderDesc
            {
                Weights = weights.BasePointer,
                WeightsLength = checked((ulong)weights.Length),
                Layers = layerPointer,
                RopeCosines = cosinePointer,
                RopeSines = sinePointer,
                TokenEmbedding = Offset(weights, layout.TokenEmbedding, WeightEncoding.GgmlQ8Zero),
                OutputNorm = Offset(weights, layout.OutputNorm, WeightEncoding.Fp32),
                Output = Offset(weights, layout.Output, WeightEncoding.GgmlQ8Zero),
                LayerCount = (uint)dimensions.LayerCount,
                Hidden = (uint)dimensions.HiddenSize,
                FeedForward = (uint)dimensions.FeedForwardSize,
                Heads = (uint)dimensions.AttentionHeads,
                KeyValueHeads = (uint)dimensions.KeyValueHeads,
                HeadDimension = (uint)dimensions.HeadDimension,
                Vocabulary = (uint)dimensions.VocabularySize,
                Context = (uint)dimensions.ContextSize,
                SessionSlots = (uint)capacity.SessionSlots,
                StepTokens = (uint)capacity.StepTokens,
                LogitsRows = (uint)capacity.LogitsRows,
                RmsEpsilon = dimensions.RmsNormEpsilon,
                MatrixEncoding = EncodingQ8Zero,
                RopeLayout = layout.RopeLayout == RotaryLayout.NeoX ? 0u : 1u,
                Activation = 0,
                KvPrecision = kvPrecision == KvCachePrecision.Fp16 ? 1u : 0u,
                KvGrowthPositions = (uint)dimensions.KvGrowthPositions,
            };
            return _library.CreateDecoder(backend, description);
        }
    }

    private static NativeDecoderLayerOffsets Offsets(IMappedWeights weights, DenseDecoderLayer layer) => new()
    {
        AttentionNorm = Offset(weights, layer.AttentionNorm, WeightEncoding.Fp32),
        Query = Offset(weights, layer.Query, WeightEncoding.GgmlQ8Zero),
        Key = Offset(weights, layer.Key, WeightEncoding.GgmlQ8Zero),
        Value = Offset(weights, layer.Value, WeightEncoding.GgmlQ8Zero),
        QueryBias = Optional(weights, layer.QueryBias),
        KeyBias = Optional(weights, layer.KeyBias),
        ValueBias = Optional(weights, layer.ValueBias),
        QueryNorm = NoTensor,
        KeyNorm = NoTensor,
        AttentionOutput = Offset(weights, layer.AttentionOutput, WeightEncoding.GgmlQ8Zero),
        FeedForwardNorm = Offset(weights, layer.FeedForwardNorm, WeightEncoding.Fp32),
        Gate = Offset(weights, layer.Gate, WeightEncoding.GgmlQ8Zero),
        Up = Offset(weights, layer.Up, WeightEncoding.GgmlQ8Zero),
        Down = Offset(weights, layer.Down, WeightEncoding.GgmlQ8Zero),
    };

    private static ulong Optional(IMappedWeights weights, DecoderWeight? weight) =>
        weight is { } present ? Offset(weights, present, WeightEncoding.Fp32) : NoTensor;

    private static ulong Offset(IMappedWeights weights, DecoderWeight weight, WeightEncoding expected)
    {
        if (!string.Equals(weight.Source.File, weights.SourceFile, StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"GPU backends read one mapped file; '{weight.Source.File}' differs from '{weights.SourceFile}'.");
        }

        return weight.Encoding == expected
            ? checked((ulong)weight.Source.Offset)
            : throw new NotSupportedException(
                $"GPU backends read this tensor as {expected}; the package stores {weight.Encoding}.");
    }

    /// <summary>Cosines and sines for every position, from the same frequencies every CPU backend uses.</summary>
    private static (float[] Cosines, float[] Sines) BuildRopeTables(DecoderDimensions dimensions)
    {
        var frequencies = new RopeFrequencies(dimensions.HeadDimension, dimensions.RopeTheta, dimensions.RopeScaling);
        var half = frequencies.Half;
        var cosines = new float[checked(dimensions.ContextSize * half)];
        var sines = new float[cosines.Length];
        _ = Parallel.For(0, dimensions.ContextSize, position =>
            frequencies.Compute(position, cosines.AsSpan(position * half, half), sines.AsSpan(position * half, half)));
        return (cosines, sines);
    }

    private static string Slug(string name) =>
        string.Join('-', name.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Where(part => !string.Equals(part, "apple", StringComparison.Ordinal)));

    static GpuDecoderExecutor()
    {
        if (Unsafe.SizeOf<BatchToken>() != 16 || sizeof(NativeDecoderDesc) != 136 || sizeof(NativeGpuDeviceInfo) != 152)
        {
            throw new PlatformNotSupportedException("GPU ABI struct layouts differ from the native library.");
        }
    }
}
