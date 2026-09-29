using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// Hyperparameters of a pre-norm dense decoder plus the instance context bound. A family adapter reads them
/// from its source format; nothing here depends on a model brand.
/// </summary>
internal sealed record DecoderDimensions(
    int LayerCount,
    int HiddenSize,
    int FeedForwardSize,
    int AttentionHeads,
    int KeyValueHeads,
    int VocabularySize,
    int ContextSize,
    float RopeTheta,
    float RmsNormEpsilon,
    RopeScaling? RopeScaling = null,
    ModelLoading.KvPageActivation? KvPages = null,
    int KvGrowthPositions = 1024)
{
    public int HeadDimension => HiddenSize / AttentionHeads;

    public int KvWidth => KeyValueHeads * HeadDimension;
}

/// <summary>
/// One backend's direct single-session evaluation; positions always start at zero for a prompt. Callers hold
/// the model's execution gate.
/// </summary>
internal interface IDecoderExecutor : IDisposable
{
    string RuntimeProfile { get; }

    string KernelImplementation { get; }

    /// <summary>Evaluates prompt positions <c>0..n-1</c> and returns the final position's logits.</summary>
    ReadOnlyMemory<float> Prefill(IReadOnlyList<int> tokens, Action<int>? evaluated = null) =>
        PrefillFrom(tokens, 0, evaluated);

    /// <summary>
    /// Evaluates positions <paramref name="start"/>..n-1 of the direct slot, whose K and V before <paramref name="start"/>
    /// already hold the same tokens (ADR-018), and returns the final position's logits.
    /// </summary>
    ReadOnlyMemory<float> PrefillFrom(IReadOnlyList<int> tokens, int start, Action<int>? evaluated);

    /// <summary>Evaluates one token at <paramref name="position"/> and returns its logits.</summary>
    ReadOnlyMemory<float> Decode(int token, int position);

    /// <summary>Bytes of KV currently allocated across every slot (ADR-017).</summary>
    long AllocatedKvBytes { get; }

    /// <summary>Sizes the direct slot for <paramref name="positions"/> at once, so a known request does not grow it.</summary>
    void Reserve(int positions);
}

/// <summary>A weight read in place: its source byte range (Model IR) and physical encoding.</summary>
internal readonly record struct DecoderWeight(WeightSourceRange Source, WeightEncoding Encoding);

/// <summary>One pre-norm block. Absent tensors (for example Llama's attention biases) are <see langword="null"/>.</summary>
internal sealed record DenseDecoderLayer(
    DecoderWeight AttentionNorm,
    DecoderWeight Query,
    DecoderWeight Key,
    DecoderWeight Value,
    DecoderWeight? QueryBias,
    DecoderWeight? KeyBias,
    DecoderWeight? ValueBias,
    DecoderWeight AttentionOutput,
    DecoderWeight FeedForwardNorm,
    DecoderWeight Gate,
    DecoderWeight Up,
    DecoderWeight Down);

/// <summary>
/// A format- and brand-neutral dense decoder: RMS norms, grouped-query attention with RoPE, and a SwiGLU
/// feed-forward. A family adapter maps its source tensors onto this layout; GPU backends consume only this.
/// </summary>
/// <param name="Architecture">Canonical family name recorded in runtime profiles and evidence.</param>
/// <param name="Dimensions">Validated hyperparameters and context bound.</param>
/// <param name="RopeLayout">Rotary coordinate pairing.</param>
/// <param name="TokenEmbedding">Vocabulary-by-hidden embedding table.</param>
/// <param name="OutputNorm">Final RMS norm weights.</param>
/// <param name="Output">Vocabulary projection (may alias the embedding for tied models).</param>
/// <param name="Layers">Blocks in execution order.</param>
internal sealed record DenseDecoderLayout(
    string Architecture,
    DecoderDimensions Dimensions,
    RotaryLayout RopeLayout,
    DecoderWeight TokenEmbedding,
    DecoderWeight OutputNorm,
    DecoderWeight Output,
    IReadOnlyList<DenseDecoderLayer> Layers);

/// <summary>A read-only mapping of the package file that a layout's source ranges point into.</summary>
internal unsafe interface IMappedWeights
{
    /// <summary>Package-relative file name that every <see cref="WeightSourceRange.File"/> must match.</summary>
    string SourceFile { get; }

    /// <summary>Bytes in the mapped file.</summary>
    long Length { get; }

    /// <summary>Page-aligned address of file offset zero; valid until the mapping is disposed.</summary>
    byte* BasePointer { get; }
}

/// <summary>Step and slot bounds derived from <c>ModelLoadOptions</c> (ADR-007).</summary>
internal sealed record DecoderStepCapacity(int StepTokens, int PrefillTokens, int SessionSlots, int LogitsRows)
{
    /// <summary>Logit rows cover both concurrent sessions and one scoring step (ADR-015); a step holds either.</summary>
    public static DecoderStepCapacity Create(int prefillChunkTokens, int contextSize, int maximumSessions, int scoringRows)
    {
        var logitsRows = Math.Max(maximumSessions, scoringRows);
        return new(
            Math.Max(Math.Min(prefillChunkTokens, contextSize), logitsRows),
            Math.Min(prefillChunkTokens, contextSize),
            maximumSessions + 1,
            logitsRows);
    }
}
