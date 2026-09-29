using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Per-step KV page masks (ADR-016). A decode unit is a token before the step's prompt boundary whose slot has no
/// other token in the step; each unit and KV head gets the positions <see cref="KvPageSelector"/> selects. Prompt tokens
/// stay dense even alone in a step, so chunking and prefix reuse never change prompt numerics.
/// </summary>
internal sealed class Qwen2KvPageMasks(
    KvPageActivation activation,
    DecoderDimensions dimensions,
    Qwen2KvSlots slots,
    BatchToken[] batch,
    float[] queries)
{
    private readonly bool[]?[] _masks = new bool[]?[slots.Count];
    private readonly int[] _unitSlot = new int[batch.Length];

    /// <summary>Selects positions for every decode unit of the first <paramref name="count"/> tokens.</summary>
    public void Prepare(int layer, int count, int promptStart)
    {
        var headDimension = dimensions.HeadDimension;
        var group = dimensions.AttentionHeads / dimensions.KeyValueHeads;
        for (var token = 0; token < count; token++)
        {
            var slot = batch[token].Slot;
            var dense = token >= promptStart;
            for (var other = 0; other < count && !dense; other++)
            {
                dense = other != token && batch[other].Slot == slot;
            }

            _unitSlot[token] = dense ? -1 : slot;
            if (dense)
            {
                continue;
            }

            var mask = _masks[slot] ??= new bool[dimensions.KeyValueHeads * dimensions.ContextSize];
            var position = batch[token].Position;
            var cache = slots[slot];
            for (var keyValueHead = 0; keyValueHead < dimensions.KeyValueHeads; keyValueHead++)
            {
                var head = keyValueHead;
                var headQueries = queries.AsSpan(
                    (token * dimensions.HiddenSize) + (keyValueHead * group * headDimension), group * headDimension);
                KvPageSelector.Select(activation, headQueries, group, headDimension,
                    cached => cache.GetKey(layer, cached, head, headDimension), position,
                    unchecked((activation.Seed * 1_000_003) + (layer * 7_919) + (keyValueHead * 104_729) + position),
                    mask.AsSpan(keyValueHead * dimensions.ContextSize, position + 1));
            }
        }
    }

    /// <summary>The attended positions of token <paramref name="token"/> and KV head, or empty for dense attention.</summary>
    public ReadOnlySpan<bool> For(int token, int keyValueHead, int length) =>
        _unitSlot[token] is var slot and >= 0
            ? _masks[slot].AsSpan(keyValueHead * dimensions.ContextSize, length)
            : default;
}
