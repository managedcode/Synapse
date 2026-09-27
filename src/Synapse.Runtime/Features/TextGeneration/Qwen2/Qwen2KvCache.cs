namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed class Qwen2KvCache
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly int _kvWidth;

    public Qwen2KvCache(int layers, int capacity, int kvWidth)
    {
        Capacity = capacity;
        _kvWidth = kvWidth;
        _keys = new float[layers][];
        _values = new float[layers][];
        for (var layer = 0; layer < layers; layer++)
        {
            _keys[layer] = new float[checked(capacity * kvWidth)];
            _values[layer] = new float[checked(capacity * kvWidth)];
        }
    }

    public int Capacity { get; }

    public void Store(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if ((uint)position >= (uint)Capacity || key.Length != _kvWidth || value.Length != _kvWidth)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        key.CopyTo(_keys[layer].AsSpan(position * _kvWidth, _kvWidth));
        value.CopyTo(_values[layer].AsSpan(position * _kvWidth, _kvWidth));
    }

    public ReadOnlySpan<float> GetKey(int layer, int position, int head, int headDimension) => _keys[layer].AsSpan((position * _kvWidth) + (head * headDimension), headDimension);

    public ReadOnlySpan<float> GetValue(int layer, int position, int head, int headDimension) => _values[layer].AsSpan((position * _kvWidth) + (head * headDimension), headDimension);
}
