namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed class Qwen2Scratch(int hiddenSize, int kvWidth, int feedForwardSize, int vocabularySize, int contextSize)
{
    public float[] Hidden { get; } = new float[hiddenSize];
    public float[] Residual { get; } = new float[hiddenSize];
    public float[] Normalized { get; } = new float[hiddenSize];
    public float[] Query { get; } = new float[hiddenSize];
    public float[] Key { get; } = new float[kvWidth];
    public float[] Value { get; } = new float[kvWidth];
    public float[] Attention { get; } = new float[hiddenSize];
    public float[] Projection { get; } = new float[hiddenSize];
    public float[] Gate { get; } = new float[feedForwardSize];
    public float[] Up { get; } = new float[feedForwardSize];
    public float[] Logits { get; } = new float[vocabularySize];
    public float[] Scores { get; } = new float[contextSize];
}
