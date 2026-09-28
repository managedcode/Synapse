namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// Computes <c>output[t * stride + (r - rowStart)] = dot(W[r], A[t])</c> for Q8_0 weight rows and
/// Q8_0 activation tokens. Integer block dots are exact; scale products and cross-block sums are FP32.
/// </summary>
internal abstract unsafe class Q8MatrixKernel
{
    public abstract string Name { get; }

    public void Multiply(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride)
    {
        ArgumentNullException.ThrowIfNull(activations);
        ArgumentNullException.ThrowIfNull(output);
        if (rowStart < 0 || rowCount <= 0 || rowStart > matrix.Rows - rowCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rowCount),
                $"Rows [{rowStart}, {rowStart + rowCount}) exceed matrix rows {matrix.Rows}.");
        }

        if (tokenCount <= 0 || tokenCount > activations.Capacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tokenCount),
                $"Token count {tokenCount} is outside activation capacity {activations.Capacity}.");
        }

        if (activations.Columns != matrix.Columns)
        {
            throw new ArgumentException(
                $"Activation width {activations.Columns} does not match matrix width {matrix.Columns}.");
        }

        if (tokenCount > 1 && outputStride < rowCount)
        {
            throw new ArgumentOutOfRangeException(nameof(outputStride));
        }

        MultiplyCore(matrix, rowStart, rowCount, activations, tokenCount, output, outputStride);
    }

    protected abstract void MultiplyCore(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride);
}
