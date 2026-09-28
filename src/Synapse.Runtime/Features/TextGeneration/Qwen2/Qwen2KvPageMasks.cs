using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Per-step KV page masks (ADR-016). A decode unit is a token whose slot has no other token in the step; each unit
/// and KV head gets the positions <see cref="KvPageSelector"/> selects. Prompt runs stay dense.
/// </summary>
internal sealed class Qwen2KvPageMasks
{
    private readonly KvPageActivation _activation;
    private readonly DecoderDimensions _dimensions;
    private readonly Qwen2KvSlots _slots;
    private readonly BatchToken[] _batch;
    private readonly float[] _queries;
    private readonly bool[]?[] _masks;
    private readonly int[] _unitSlot;

    public Qwen2KvPageMasks(
        KvPageActivation activation,
        DecoderDimensions dimensions,
        Qwen2KvSlots slots,
        BatchToken[] batch,
        float[] queries)
    {
        _activation = activation;
        _dimensions = dimensions;
        _slots = slots;
        _batch = batch;
        _queries = queries;
        _masks = new bool[]?[slots.Count];
        _unitSlot = new int[batch.Length];
    }

    /// <summary>Selects positions for every decode unit of the first <paramref name="count"/> tokens.</summary>
    public void Prepare(int layer, int count)
    {
        var headDimension = _dimensions.HeadDimension;
        var group = _dimensions.AttentionHeads / _dimensions.KeyValueHeads;
        for (var token = 0; token < count; token++)
        {
            var slot = _batch[token].Slot;
            var shared = false;
            for (var other = 0; other < count && !shared; other++)
            {
                shared = other != token && _batch[other].Slot == slot;
            }

            _unitSlot[token] = shared ? -1 : slot;
            if (shared)
            {
                continue;
            }

            var mask = _masks[slot] ??= new bool[_dimensions.KeyValueHeads * _dimensions.ContextSize];
            var position = _batch[token].Position;
            var cache = _slots[slot];
            for (var keyValueHead = 0; keyValueHead < _dimensions.KeyValueHeads; keyValueHead++)
            {
                var head = keyValueHead;
                var queries = _queries.AsSpan(
                    (token * _dimensions.HiddenSize) + (keyValueHead * group * headDimension), group * headDimension);
                KvPageSelector.Select(_activation, queries, group, headDimension,
                    cached => cache.GetKey(layer, cached, head, headDimension), position,
                    unchecked((_activation.Seed * 1_000_003) + (layer * 7_919) + (keyValueHead * 104_729) + position),
                    mask.AsSpan(keyValueHead * _dimensions.ContextSize, position + 1));
            }
        }
    }

    /// <summary>The attended positions of token <paramref name="token"/> and KV head, or empty for dense attention.</summary>
    public ReadOnlySpan<bool> For(int token, int keyValueHead, int length) =>
        _unitSlot[token] is var slot and >= 0
            ? _masks[slot].AsSpan(keyValueHead * _dimensions.ContextSize, length)
            : default;
}
