namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// FP32 K and V per layer in <c>[position][kv width]</c> order. Capacity grows on demand in multiples of
/// <c>growthPositions</c>, at least doubling, up to <c>maximumPositions</c> (ADR-017); the layout does not depend on
/// capacity, so growth only resizes.
/// </summary>
internal sealed class Qwen2KvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _kvWidth;
    private readonly int _growthPositions;

    public Qwen2KvCache(int layers, int maximumPositions, int kvWidth, int growthPositions)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(growthPositions);
        MaximumPositions = maximumPositions;
        _kvWidth = kvWidth;
        _growthPositions = growthPositions;
        _keys = new float[layers][];
        _values = new float[layers][];
        for (var layer = 0; layer < layers; layer++)
        {
            _keys[layer] = [];
            _values[layer] = [];
        }
    }

    /// <summary>Positions the cache may hold: the instance context.</summary>
    public int MaximumPositions { get; }

    /// <summary>Positions currently allocated.</summary>
    public int AllocatedPositions { get; private set; }

    /// <summary>Bytes of K and V currently allocated across layers.</summary>
    public long AllocatedBytes => 2L * _keys.Length * AllocatedPositions * _kvWidth * sizeof(float);

    public void Store(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if ((uint)position >= (uint)MaximumPositions || key.Length != _kvWidth || value.Length != _kvWidth)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        if (position >= AllocatedPositions)
        {
            Grow(position + 1);
        }

        key.CopyTo(_keys[layer].AsSpan(position * _kvWidth, _kvWidth));
        value.CopyTo(_values[layer].AsSpan(position * _kvWidth, _kvWidth));
    }

    /// <summary>Allocates exactly the positions a known request needs, rounded up to the growth unit (ADR-017).</summary>
    public void Reserve(int positions)
    {
        if (positions > AllocatedPositions)
        {
            Resize((int)Math.Min(MaximumPositions, ((long)positions + _growthPositions - 1) / _growthPositions * _growthPositions));
        }
    }

    public ReadOnlySpan<float> GetKey(int layer, int position, int head, int headDimension) => _keys[layer].AsSpan((position * _kvWidth) + (head * headDimension), headDimension);

    public ReadOnlySpan<float> GetValue(int layer, int position, int head, int headDimension) => _values[layer].AsSpan((position * _kvWidth) + (head * headDimension), headDimension);

    /// <summary>ADR-017 policy: round the need up to the growth unit, at least double, never exceed the maximum.</summary>
    internal static int NextCapacity(int needed, int current, int growth, int maximum)
    {
        var rounded = (int)Math.Min(maximum, ((long)needed + growth - 1) / growth * growth);
        return Math.Min(maximum, Math.Max(rounded, Math.Min(maximum, current * 2)));
    }

    private void Grow(int needed) => Resize(NextCapacity(needed, AllocatedPositions, _growthPositions, MaximumPositions));

    /// <summary>Grows every layer first and publishes the capacity last, so a failed allocation leaves a usable cache.</summary>
    private void Resize(int positions)
    {
        var length = checked(positions * _kvWidth);
        for (var layer = 0; layer < _keys.Length; layer++)
        {
            Array.Resize(ref _keys[layer], Math.Max(length, _keys[layer].Length));
            Array.Resize(ref _values[layer], Math.Max(length, _values[layer].Length));
        }

        AllocatedPositions = positions;
    }
}
