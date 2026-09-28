using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal static class Qwen2Vectors
{
    /// <summary>Per-worker partial-result floats; a region over T tokens takes at most this / T rows per chunk.</summary>
    public const int PartialCapacity = 16_384;

    public static float[][] PerWorker(int threadCount, int length)
    {
        var buffers = new float[threadCount][];
        for (var worker = 0; worker < threadCount; worker++)
        {
            buffers[worker] = GC.AllocateArray<float>(length, pinned: true);
        }

        return buffers;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddInPlace(Span<float> destination, ReadOnlySpan<float> values)
    {
        var index = 0;
        for (; index <= destination.Length - Vector128<float>.Count; index += Vector128<float>.Count)
        {
            (Vector128.Create(destination[index..]) + Vector128.Create(values[index..])).CopyTo(destination[index..]);
        }

        for (; index < destination.Length; index++)
        {
            destination[index] += values[index];
        }
    }
}

/// <summary>Per-session KV slots, allocated on first use. Slot 0 serves the direct synchronous path.</summary>
internal sealed class Qwen2KvSlots(DecoderDimensions dimensions, int slotCount)
{
    private readonly Qwen2KvCache?[] _slots = new Qwen2KvCache?[slotCount];

    public int Count => _slots.Length;

    public Qwen2KvCache this[int slot] =>
        _slots[slot] ??= new Qwen2KvCache(dimensions.LayerCount, dimensions.ContextSize, dimensions.KvWidth);
}
