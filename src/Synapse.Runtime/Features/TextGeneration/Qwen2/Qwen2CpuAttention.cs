using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>
/// Causal grouped-query attention over FP32 KV slots for one batched step. Each (token, head) pair reads only its
/// own slot up to its own position, so pairs are independent: large steps run on the worker pool and small
/// decode steps stay on the caller.
/// </summary>
internal sealed class Qwen2CpuAttention
{
    private const long ParallelMultiplyAdds = 1 << 15;
    private readonly Qwen2Dimensions _dimensions;
    private readonly Qwen2KvSlots _slots;
    private readonly BatchToken[] _batch;
    private readonly Qwen2CpuScratch _scratch;
    private readonly CpuWorkerPool _pool;
    private readonly float[][] _scores;
    private readonly AttentionWork _work;
    private readonly float _scale;
    private int _layer;

    public Qwen2CpuAttention(
        Qwen2Dimensions dimensions,
        Qwen2KvSlots slots,
        BatchToken[] batch,
        Qwen2CpuScratch scratch,
        CpuWorkerPool pool)
    {
        _dimensions = dimensions;
        _slots = slots;
        _batch = batch;
        _scratch = scratch;
        _pool = pool;
        _scale = 1.0f / MathF.Sqrt(dimensions.HeadDimension);
        _scores = new float[pool.ThreadCount][];
        for (var worker = 0; worker < _scores.Length; worker++)
        {
            _scores[worker] = new float[dimensions.ContextSize];
        }

        _work = new AttentionWork(this);
    }

    /// <summary>Attends the first <paramref name="count"/> batch tokens after their KV writes.</summary>
    public void Execute(int layer, int count)
    {
        _layer = layer;
        var items = count * _dimensions.AttentionHeads;
        long attendedPositions = 0;
        for (var index = 0; index < count; index++)
        {
            attendedPositions += _batch[index].Position + 1;
        }

        var multiplyAdds = attendedPositions * _dimensions.AttentionHeads * _dimensions.HeadDimension;
        if (_pool.ThreadCount == 1 || multiplyAdds < ParallelMultiplyAdds)
        {
            for (var item = 0; item < items; item++)
            {
                Attend(0, item);
            }

            return;
        }

        _work.Prepare(items, _pool.ThreadCount);
        _pool.Run(_work);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Attend(int worker, int item)
    {
        var heads = _dimensions.AttentionHeads;
        var headDimension = _dimensions.HeadDimension;
        var token = item / heads;
        var head = item % heads;
        var keyValueHead = head / (heads / _dimensions.KeyValueHeads);
        var entry = _batch[token];
        var position = entry.Position;
        var cache = _slots[entry.Slot];
        var offset = (token * _dimensions.HiddenSize) + (head * headDimension);
        ReadOnlySpan<float> query = _scratch.Query.AsSpan(offset, headDimension);
        var scores = _scores[worker].AsSpan(0, position + 1);
        for (var cached = 0; cached <= position; cached++)
        {
            scores[cached] = Dot(query, cache.GetKey(_layer, cached, keyValueHead, headDimension)) * _scale;
        }

        Softmax(scores);
        var destination = _scratch.Attention.AsSpan(offset, headDimension);
        destination.Clear();
        for (var cached = 0; cached <= position; cached++)
        {
            MultiplyAdd(destination, scores[cached], cache.GetValue(_layer, cached, keyValueHead, headDimension));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        var sum = Vector128<float>.Zero;
        var index = 0;
        for (; index <= left.Length - Vector128<float>.Count; index += Vector128<float>.Count)
        {
            sum += Vector128.Create(left[index..]) * Vector128.Create(right[index..]);
        }

        var total = Vector128.Sum(sum);
        for (; index < left.Length; index++)
        {
            total += left[index] * right[index];
        }

        return total;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyAdd(Span<float> destination, float weight, ReadOnlySpan<float> values)
    {
        var scale = Vector128.Create(weight);
        var index = 0;
        for (; index <= destination.Length - Vector128<float>.Count; index += Vector128<float>.Count)
        {
            (Vector128.Create(destination[index..]) + (scale * Vector128.Create(values[index..])))
                .CopyTo(destination[index..]);
        }

        for (; index < destination.Length; index++)
        {
            destination[index] += weight * values[index];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Softmax(Span<float> values)
    {
        var maximum = values[0];
        for (var index = 1; index < values.Length; index++)
        {
            maximum = MathF.Max(maximum, values[index]);
        }

        var sum = 0.0f;
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = MathF.Exp(values[index] - maximum);
            sum += values[index];
        }

        var inverse = 1.0f / sum;
        for (var index = 0; index < values.Length; index++)
        {
            values[index] *= inverse;
        }
    }

    private sealed class AttentionWork(Qwen2CpuAttention owner) : ChunkedRowWork
    {
        public void Prepare(int items, int threadCount) =>
            Reset(items, threadCount, alignment: 1, minimumChunkRows: 1);

        protected override void ProcessRows(int worker, int rowStart, int rowCount)
        {
            for (var item = rowStart; item < rowStart + rowCount; item++)
            {
                owner.Attend(worker, item);
            }
        }
    }
}
