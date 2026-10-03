using System.Buffers.Binary;
using System.Globalization;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

/// <summary>
/// Registry of verified source-encoding decoders for GGUF tensor types and SafeTensors dtypes.
/// Every decoder produces the FP32 reference values that on-the-fly quantization and
/// execution consume. Types without a verified decoder fail explicitly (INV-007).
/// </summary>
public static class SourceEncodings
{
    // Indexed by ggml_type; verified against ggml_type_name from the pinned native ggml.
    private static readonly string[] GgufNames =
    [
        "f32", "f16", "q4_0", "q4_1", "DEPRECATED", "DEPRECATED", "q5_0", "q5_1", "q8_0", "q8_1",
        "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s",
        "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "i8", "i16", "i32", "i64", "f64", "iq1_m",
        "bf16", "REMOVED", "REMOVED", "REMOVED", "tq1_0", "tq2_0", "REMOVED", "REMOVED", "REMOVED", "mxfp4",
        "nvfp4", "q1_0",
    ];

    /// <summary>IEEE 754 single precision.</summary>
    public static ISourceTensorDecoder Fp32 { get; } = new FloatSourceDecoder("f32", FloatSourceKind.Fp32, sizeof(float));

    /// <summary>IEEE 754 double precision, narrowed to FP32 when representable.</summary>
    public static ISourceTensorDecoder Fp64 { get; } = Scalar("f64", 8, bytes => (float)BinaryPrimitives.ReadDoubleLittleEndian(bytes));

    /// <summary>IEEE 754 half precision.</summary>
    public static ISourceTensorDecoder Fp16 { get; } = new FloatSourceDecoder("f16", FloatSourceKind.Fp16, sizeof(ushort));

    /// <summary>Brain floating point: the upper 16 bits of an FP32 value.</summary>
    public static ISourceTensorDecoder Bf16 { get; } = new FloatSourceDecoder("bf16", FloatSourceKind.Bf16, sizeof(ushort));

    /// <summary>OCP FP8 E4M3FN: no infinities; S.1111.111 is NaN; maximum 448.</summary>
    public static ISourceTensorDecoder Fp8E4M3Fn { get; } = Scalar("f8_e4m3fn", 1, bytes => DecodeE4M3Fn(bytes[0]));

    /// <summary>OCP FP8 E5M2: the upper byte of an IEEE half-precision value.</summary>
    public static ISourceTensorDecoder Fp8E5M2 { get; } =
        Scalar("f8_e5m2", 1, bytes => (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[0] << 8)));

    // Declared after the scalar decoders: static initializers run in text order, and the table reads them.
    private static readonly Dictionary<uint, ISourceTensorDecoder> GgufDecoders = CreateGgufDecoders();

    /// <summary>ggml's name for a GGUF type identifier.</summary>
    public static string GetGgufTypeName(uint type) =>
        type < GgufNames.Length ? GgufNames[type] : string.Create(CultureInfo.InvariantCulture, $"unknown({type})");

    /// <summary>Returns the verified decoder for a GGUF type, when one exists.</summary>
    public static bool TryGetGguf(uint type, out ISourceTensorDecoder? decoder) =>
        GgufDecoders.TryGetValue(type, out decoder);

    /// <summary>Returns the verified decoder for a GGUF type or throws <see cref="SourceEncodingFailure.UnsupportedType"/>.</summary>
    public static ISourceTensorDecoder GetGguf(uint type) =>
        TryGetGguf(type, out var decoder)
            ? decoder!
            : throw new SourceEncodingException(
                SourceEncodingFailure.UnsupportedType,
                $"GGUF tensor type {type} ({GetGgufTypeName(type)}) has no verified Synapse decoder.");

    /// <summary>Returns the decoder for a SafeTensors dtype, when one exists.</summary>
    public static bool TryGetSafeTensors(string dtype, out ISourceTensorDecoder? decoder)
    {
        decoder = dtype switch
        {
            "F32" => Fp32,
            "F64" => Fp64,
            "F16" => Fp16,
            "BF16" => Bf16,
            "F8_E4M3" => Fp8E4M3Fn,
            "F8_E5M2" => Fp8E5M2,
            _ => null,
        };
        return decoder is not null;
    }

    private static Dictionary<uint, ISourceTensorDecoder> CreateGgufDecoders()
    {
        var decoders = new Dictionary<uint, ISourceTensorDecoder>
        {
            [(uint)GgufType.F32] = Fp32,
            [(uint)GgufType.F16] = Fp16,
            [(uint)GgufType.BF16] = Bf16,
            [(uint)GgufType.F64] = Fp64,
        };
        foreach (var (type, decoder) in GgufLegacyBlocks.Create().Concat(GgufKQuantBlocks.Create()).Concat(GgufTernaryBlocks.Create()))
        {
            decoders.Add((uint)type, decoder);
        }

        return decoders;
    }

    private static BlockSourceDecoder Scalar(string id, int bytes, ScalarReader read) =>
        new(
            id,
            1,
            bytes,
            block => float.IsFinite(read(block)) ? null : SourceEncodingFailure.NonFiniteValue,
            (block, output) => output[0] = read(block));

    private delegate float ScalarReader(ReadOnlySpan<byte> bytes);

    private static float DecodeE4M3Fn(byte code)
    {
        var exponent = (code >> 3) & 0x0F;
        var mantissa = code & 0x07;
        var magnitude = exponent == 0x0F && mantissa == 0x07
            ? float.NaN
            : exponent == 0
                ? MathF.ScaleB(mantissa, -9)
                : MathF.ScaleB(8 + mantissa, exponent - 10);
        return (code & 0x80) != 0 ? -magnitude : magnitude;
    }
}
