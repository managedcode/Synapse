using System.Runtime.CompilerServices;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

/// <summary>Fused Q/K/V projection with biases; the three matrices form one row space.</summary>
internal sealed unsafe class Qwen2QkvWork(Q8MatrixKernel kernel, Qwen2CpuScratch scratch) : ChunkedRowWork
{
    private Qwen2CpuLayer _layer = null!;
    private Qwen2LayerWeights _weights = null!;
    private int _tokens;

    public void Prepare(Qwen2CpuLayer layer, Qwen2LayerWeights weights, int tokens, int threadCount)
    {
        _layer = layer;
        _weights = weights;
        _tokens = tokens;
        Reset(layer.Query.Rows + layer.Key.Rows + layer.Value.Rows, threadCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ProcessRows(int worker, int rowStart, int rowCount)
    {
        var end = rowStart + rowCount;
        var keyStart = _layer.Query.Rows;
        var valueStart = keyStart + _layer.Key.Rows;
        Project(_layer.Query, scratch.Query, _weights.QueryBias, 0, rowStart, end);
        Project(_layer.Key, scratch.Key, _weights.KeyBias, keyStart, rowStart, end);
        Project(_layer.Value, scratch.Value, _weights.ValueBias, valueStart, rowStart, end);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Project(in Q8Matrix matrix, float[] output, float[] bias, int segmentStart, int start, int end)
    {
        var from = Math.Max(start, segmentStart) - segmentStart;
        var to = Math.Min(end, segmentStart + matrix.Rows) - segmentStart;
        if (from >= to)
        {
            return;
        }

        fixed (float* destination = output)
        {
            kernel.Multiply(matrix, from, to - from, scratch.HiddenActivations, _tokens, destination + from, matrix.Rows);
        }

        for (var token = 0; token < _tokens; token++)
        {
            var values = output.AsSpan((token * matrix.Rows) + from, to - from);
            Qwen2Vectors.AddInPlace(values, bias.AsSpan(from, to - from));
        }
    }
}

/// <summary>Output or down projection whose result is added to the residual stream in place.</summary>
internal sealed unsafe class Qwen2ResidualWork(Q8MatrixKernel kernel, float[] hidden, int threadCount)
    : ChunkedRowWork
{
    private readonly float[][] _partials = Qwen2Vectors.PerWorker(threadCount, Qwen2Vectors.PartialCapacity);
    private Q8Matrix _matrix;
    private Q8ActivationBuffer _activations = null!;
    private int _tokens;

    public void Prepare(in Q8Matrix matrix, Q8ActivationBuffer activations, int tokens, int threads)
    {
        _matrix = matrix;
        _activations = activations;
        _tokens = tokens;
        Reset(matrix.Rows, threads, maximumChunkRows: Qwen2Vectors.PartialCapacity / tokens);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ProcessRows(int worker, int rowStart, int rowCount)
    {
        var partial = _partials[worker];
        fixed (float* destination = partial)
        {
            kernel.Multiply(_matrix, rowStart, rowCount, _activations, _tokens, destination, rowCount);
        }

        for (var token = 0; token < _tokens; token++)
        {
            Qwen2Vectors.AddInPlace(
                hidden.AsSpan((token * _matrix.Rows) + rowStart, rowCount),
                partial.AsSpan(token * rowCount, rowCount));
        }
    }
}

/// <summary>Gate and up projections combined by SwiGLU into the feed-forward activation rows.</summary>
internal sealed unsafe class Qwen2GateUpWork(Q8MatrixKernel kernel, Qwen2CpuScratch scratch, int threadCount)
    : ChunkedRowWork
{
    private readonly float[][] _gates = Qwen2Vectors.PerWorker(threadCount, Qwen2Vectors.PartialCapacity);
    private readonly float[][] _ups = Qwen2Vectors.PerWorker(threadCount, Qwen2Vectors.PartialCapacity);
    private Qwen2CpuLayer _layer = null!;
    private int _tokens;

    public void Prepare(Qwen2CpuLayer layer, int tokens, int threads)
    {
        _layer = layer;
        _tokens = tokens;
        Reset(layer.Gate.Rows, threads, maximumChunkRows: Qwen2Vectors.PartialCapacity / tokens);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ProcessRows(int worker, int rowStart, int rowCount)
    {
        var gate = _gates[worker];
        var up = _ups[worker];
        fixed (float* gateDestination = gate)
        fixed (float* upDestination = up)
        {
            kernel.Multiply(_layer.Gate, rowStart, rowCount, scratch.HiddenActivations, _tokens, gateDestination, rowCount);
            kernel.Multiply(_layer.Up, rowStart, rowCount, scratch.HiddenActivations, _tokens, upDestination, rowCount);
        }

        var width = _layer.Gate.Rows;
        for (var token = 0; token < _tokens; token++)
        {
            var output = scratch.FeedForward.AsSpan((token * width) + rowStart, rowCount);
            var gateRow = gate.AsSpan(token * rowCount, rowCount);
            var upRow = up.AsSpan(token * rowCount, rowCount);
            for (var index = 0; index < rowCount; index++)
            {
                var value = gateRow[index];
                output[index] = value / (1.0f + MathF.Exp(-value)) * upRow[index];
            }
        }
    }
}

/// <summary>
/// Vocabulary projection for activation rows <c>0..rows-1</c>, written directly into logits rows, so one pass over
/// the vocabulary weights serves every session that needs logits in a step.
/// </summary>
internal sealed unsafe class Qwen2LogitsWork(Q8MatrixKernel kernel, Qwen2CpuScratch scratch) : ChunkedRowWork
{
    private Q8Matrix _matrix;
    private int _rows;

    public void Prepare(in Q8Matrix matrix, int rows, int threads)
    {
        _matrix = matrix;
        _rows = rows;
        Reset(matrix.Rows, threads);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ProcessRows(int worker, int rowStart, int rowCount)
    {
        fixed (float* logits = scratch.Logits)
        {
            kernel.Multiply(_matrix, rowStart, rowCount, scratch.HiddenActivations, _rows, logits + rowStart, _matrix.Rows);
        }
    }
}
