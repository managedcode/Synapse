using System.Buffers.Binary;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

/// <summary>GGUF tensor type identifiers, as defined by ggml's <c>enum ggml_type</c>.</summary>
#pragma warning disable CA1707 // Names mirror ggml's type names so that import errors are unambiguous.
public enum GgufType : uint
{
    /// <summary>IEEE 754 single precision.</summary>
    F32 = 0,
    /// <summary>IEEE 754 half precision.</summary>
    F16 = 1,
    /// <summary>4-bit symmetric blocks of 32.</summary>
    Q4_0 = 2,
    /// <summary>4-bit affine blocks of 32.</summary>
    Q4_1 = 3,
    /// <summary>5-bit symmetric blocks of 32.</summary>
    Q5_0 = 6,
    /// <summary>5-bit affine blocks of 32.</summary>
    Q5_1 = 7,
    /// <summary>8-bit symmetric blocks of 32.</summary>
    Q8_0 = 8,
    /// <summary>8-bit blocks with sums, used for activations.</summary>
    Q8_1 = 9,
    /// <summary>2-bit k-quant super-blocks of 256.</summary>
    Q2_K = 10,
    /// <summary>3-bit k-quant super-blocks of 256.</summary>
    Q3_K = 11,
    /// <summary>4-bit k-quant super-blocks of 256.</summary>
    Q4_K = 12,
    /// <summary>5-bit k-quant super-blocks of 256.</summary>
    Q5_K = 13,
    /// <summary>6-bit k-quant super-blocks of 256.</summary>
    Q6_K = 14,
    /// <summary>8-bit k-quant super-blocks of 256.</summary>
    Q8_K = 15,
    /// <summary>Grid-coded 2-bit super-blocks.</summary>
    IQ2_XXS = 16,
    /// <summary>Grid-coded 2-bit super-blocks.</summary>
    IQ2_XS = 17,
    /// <summary>Grid-coded 3-bit super-blocks.</summary>
    IQ3_XXS = 18,
    /// <summary>Grid-coded 1-bit super-blocks.</summary>
    IQ1_S = 19,
    /// <summary>Non-linear 4-bit blocks of 32.</summary>
    IQ4_NL = 20,
    /// <summary>Grid-coded 3-bit super-blocks.</summary>
    IQ3_S = 21,
    /// <summary>Grid-coded 2-bit super-blocks.</summary>
    IQ2_S = 22,
    /// <summary>Non-linear 4-bit super-blocks of 256.</summary>
    IQ4_XS = 23,
    /// <summary>IEEE 754 double precision.</summary>
    F64 = 28,
    /// <summary>Grid-coded 1-bit super-blocks.</summary>
    IQ1_M = 29,
    /// <summary>Brain floating point.</summary>
    BF16 = 30,
    /// <summary>Ternary base-3 packed super-blocks of 256.</summary>
    TQ1_0 = 34,
    /// <summary>Ternary 2-bit packed super-blocks of 256.</summary>
    TQ2_0 = 35,
    /// <summary>OCP microscaling FP4 blocks of 32 with an E8M0 exponent.</summary>
    MXFP4 = 39,
    /// <summary>NVIDIA FP4 blocks of 64.</summary>
    NVFP4 = 40,
    /// <summary>1-bit blocks of 128.</summary>
    Q1_0 = 41,
}
#pragma warning restore CA1707

/// <summary>Decodes one stored source encoding to the FP32 reference representation.</summary>
public interface ISourceTensorDecoder
{
    /// <summary>Stable encoding identity, for example <c>gguf.q4_K</c> or <c>bf16</c>.</summary>
    string EncodingId { get; }

    /// <summary>Logical elements per stored block.</summary>
    int BlockElements { get; }

    /// <summary>Stored bytes per block.</summary>
    int BlockBytes { get; }

    /// <summary>Exact stored length of <paramref name="elements"/> values.</summary>
    long GetByteLength(long elements);

    /// <summary>Validates every block, then writes <c>destination.Length</c> values.</summary>
    void Decode(ReadOnlySpan<byte> source, Span<float> destination);
}

/// <summary>Stable categories of source-decoding failures.</summary>
public enum SourceEncodingFailure
{
    /// <summary>The container type has no verified decoder.</summary>
    UnsupportedType,
    /// <summary>A block scale is NaN, infinite, or would overflow FP32.</summary>
    InvalidScale,
    /// <summary>A stored value is NaN or infinite.</summary>
    NonFiniteValue,
}

/// <summary>A typed source-decoding failure raised before the destination is written.</summary>
/// <remarks>Creates a failure with its stable category.</remarks>
public sealed class SourceEncodingException(SourceEncodingFailure failure, string message) : FormatException(message)
{

    /// <summary>Stable failure category.</summary>
    public SourceEncodingFailure Failure { get; } = failure;
}

internal delegate SourceEncodingFailure? BlockValidator(ReadOnlySpan<byte> block);

internal delegate void BlockDecoder(ReadOnlySpan<byte> block, Span<float> output);

internal sealed class BlockSourceDecoder(
    string encodingId,
    int blockElements,
    int blockBytes,
    BlockValidator validate,
    BlockDecoder decode) : ISourceTensorDecoder
{
    public string EncodingId { get; } = encodingId;

    public int BlockElements { get; } = blockElements;

    public int BlockBytes { get; } = blockBytes;

    public long GetByteLength(long elements) =>
        elements > 0 && elements % BlockElements == 0
            ? checked(elements / BlockElements * BlockBytes)
            : throw new ArgumentException($"{EncodingId} needs a positive multiple of {BlockElements} elements.", nameof(elements));

    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (source.Length != GetByteLength(destination.Length))
        {
            throw new ArgumentException($"{EncodingId} needs {GetByteLength(destination.Length)} bytes for {destination.Length} values.");
        }

        var blocks = destination.Length / BlockElements;
        for (var block = 0; block < blocks; block++)
        {
            if (validate(source.Slice(block * BlockBytes, BlockBytes)) is { } failure)
            {
                throw new SourceEncodingException(failure, $"{EncodingId} block {block} fails validation: {failure}.");
            }
        }

        for (var block = 0; block < blocks; block++)
        {
            decode(source.Slice(block * BlockBytes, BlockBytes), destination.Slice(block * BlockElements, BlockElements));
        }
    }
}

internal static class BlockFields
{
    public static float Half(ReadOnlySpan<byte> block, int offset) =>
        (float)BinaryPrimitives.ReadHalfLittleEndian(block[offset..]);

    public static SourceEncodingFailure? FiniteHalves(ReadOnlySpan<byte> block, params ReadOnlySpan<int> offsets)
    {
        foreach (var offset in offsets)
        {
            if (!float.IsFinite(Half(block, offset)))
            {
                return SourceEncodingFailure.InvalidScale;
            }
        }

        return null;
    }
}
