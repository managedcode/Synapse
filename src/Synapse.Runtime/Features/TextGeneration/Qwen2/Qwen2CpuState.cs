using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Memory-mapped Q8_0 projection views for one transformer block.</summary>
internal sealed record Qwen2CpuLayer(
    Q8Matrix Query,
    Q8Matrix Key,
    Q8Matrix Value,
    Q8Matrix Output,
    Q8Matrix Gate,
    Q8Matrix Up,
    Q8Matrix Down)
{
    public static Qwen2CpuLayer Create(GgufFile file, Qwen2LayerWeights weights) => new(
        Q8Matrix.FromTensor(file, weights.Query),
        Q8Matrix.FromTensor(file, weights.Key),
        Q8Matrix.FromTensor(file, weights.Value),
        Q8Matrix.FromTensor(file, weights.AttentionOutput),
        Q8Matrix.FromTensor(file, weights.FeedForwardGate),
        Q8Matrix.FromTensor(file, weights.FeedForwardUp),
        Q8Matrix.FromTensor(file, weights.FeedForwardDown));
}

/// <summary>Row-major activations for one prompt chunk of up to <see cref="ChunkTokens"/> tokens.</summary>
internal sealed class Qwen2CpuScratch(DecoderDimensions dimensions, int chunkTokens, int logitsRows)
{
    public int ChunkTokens { get; } = chunkTokens;

    public int LogitsRows { get; } = logitsRows;

    public float[] Hidden { get; } = new float[chunkTokens * dimensions.HiddenSize];

    public float[] Normalized { get; } = new float[dimensions.HiddenSize];

    public float[] Query { get; } = new float[chunkTokens * dimensions.HiddenSize];

    public float[] Key { get; } = new float[chunkTokens * dimensions.KvWidth];

    public float[] Value { get; } = new float[chunkTokens * dimensions.KvWidth];

    public float[] Attention { get; } = new float[chunkTokens * dimensions.HiddenSize];

    public float[] FeedForward { get; } = new float[chunkTokens * dimensions.FeedForwardSize];

    public float[] Logits { get; } = GC.AllocateArray<float>(logitsRows * dimensions.VocabularySize, pinned: true);

    public Q8ActivationBuffer HiddenActivations { get; } = new(dimensions.HiddenSize, chunkTokens);

    public Q8ActivationBuffer FeedForwardActivations { get; } = new(dimensions.FeedForwardSize, chunkTokens);
}

/// <summary>
/// NeoX rotary cosines and sines cached per position on first use. Values come from the shared
/// <see cref="RopeFrequencies"/>, so rotated values are identical to the reference operator.
/// </summary>
internal sealed class Qwen2RopeTable
{
    private readonly RopeFrequencies _frequencies;
    private readonly float[] _cosines;
    private readonly float[] _sines;
    private readonly bool[] _ready;
    private readonly int _headDimension;
    private readonly int _half;

    public Qwen2RopeTable(RopeFrequencies frequencies, int contextSize)
    {
        _frequencies = frequencies;
        _half = frequencies.Half;
        _headDimension = _half * 2;
        if (_half % Vector128<float>.Count != 0)
        {
            throw new NotSupportedException($"RoPE half dimension {_half} is not a multiple of {Vector128<float>.Count}.");
        }

        _cosines = new float[checked(contextSize * _half)];
        _sines = new float[checked(contextSize * _half)];
        _ready = new bool[contextSize];
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Apply(Span<float> values, int heads, int position)
    {
        EnsurePosition(position);
        var cosines = _cosines.AsSpan(position * _half, _half);
        var sines = _sines.AsSpan(position * _half, _half);
        for (var head = 0; head < heads; head++)
        {
            var first = values.Slice(head * _headDimension, _half);
            var second = values.Slice((head * _headDimension) + _half, _half);
            for (var index = 0; index <= _half - Vector128<float>.Count; index += Vector128<float>.Count)
            {
                var x = Vector128.Create(first[index..]);
                var y = Vector128.Create(second[index..]);
                var cosine = Vector128.Create(cosines[index..]);
                var sine = Vector128.Create(sines[index..]);
                ((x * cosine) - (y * sine)).CopyTo(first[index..]);
                ((x * sine) + (y * cosine)).CopyTo(second[index..]);
            }
        }
    }

    private void EnsurePosition(int position)
    {
        if (_ready[position])
        {
            return;
        }

        _frequencies.Compute(
            position,
            _cosines.AsSpan(position * _half, _half),
            _sines.AsSpan(position * _half, _half));
        _ready[position] = true;
    }
}
