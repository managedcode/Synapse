using System.Buffers.Binary;
using System.Runtime.InteropServices;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

/// <summary>
/// Differential tests against ggml's own <c>dequantize_row_*</c> reference functions from the
/// native library that ships with the LLamaSharp benchmark subject. ggml is the format's
/// reference implementation; it is used here only as a test oracle, never at runtime.
/// </summary>
[NotInParallel]
public sealed class GgmlReferenceDecoderTests
{
    private const int Blocks = 16;

    [Test]
    public async Task BlockGeometryMatchesGgml()
    {
        var ggml = GgmlLibrary.Instance;
        foreach (var type in ReferenceCases().Select(item => item.Type))
        {
            var decoder = SourceEncodings.GetGguf((uint)type);
            await Assert.That(ggml.BlockElements(type)).IsEqualTo(decoder.BlockElements);
            await Assert.That(ggml.BlockBytes(type)).IsEqualTo(decoder.BlockBytes);
            await Assert.That(ggml.TypeName(type)).IsEqualTo(SourceEncodings.GetGgufTypeName((uint)type));
        }
    }

    [Test]
    public async Task EveryImplementedQuantMatchesGgmlDequantize()
    {
        var ggml = GgmlLibrary.Instance;
        foreach (var (type, function, affine, scales) in ReferenceCases())
        {
            var decoder = SourceEncodings.GetGguf((uint)type);
            var source = RandomBlocks(decoder, scales, seed: (int)type);
            var elements = Blocks * decoder.BlockElements;
            var expected = ggml.Dequantize(function, source, elements);
            var actual = new float[elements];

            decoder.Decode(source, actual);

            var tolerance = affine ? MathF.ScaleB(expected.Max(MathF.Abs), -20) : 0f;
            var worst = actual.Zip(expected).Max(pair => Difference(pair.First, pair.Second));
            await Assert.That(worst).IsLessThanOrEqualTo(tolerance).Because($"{type} max difference {worst}");
        }
    }

    [Test]
    public async Task HalfAndBrainFloatRowsMatchGgml()
    {
        var ggml = GgmlLibrary.Instance;
        var random = new Random(7);
        var halves = new byte[512];
        for (var index = 0; index < halves.Length; index += 2)
        {
            BinaryPrimitives.WriteHalfLittleEndian(halves.AsSpan(index), (Half)((random.NextDouble() * 2000) - 1000));
        }

        var expectedHalf = ggml.Dequantize("ggml_fp16_to_fp32_row", halves, 256);
        var expectedBrain = ggml.Dequantize("ggml_bf16_to_fp32_row", halves, 256);
        var actualHalf = new float[256];
        var actualBrain = new float[256];

        SourceEncodings.Fp16.Decode(halves, actualHalf);
        SourceEncodings.Bf16.Decode(halves, actualBrain);

        await Assert.That(actualHalf).IsEquivalentTo(expectedHalf);
        await Assert.That(actualBrain).IsEquivalentTo(expectedBrain);
    }

    private static IEnumerable<(GgufType Type, string Function, bool Affine, (int Offset, ScaleKind Kind)[] Scales)> ReferenceCases() =>
    [
        (GgufType.Q4_0, "dequantize_row_q4_0", false, [(0, ScaleKind.Half)]),
        (GgufType.Q4_1, "dequantize_row_q4_1", true, [(0, ScaleKind.Half), (2, ScaleKind.Half)]),
        (GgufType.Q5_0, "dequantize_row_q5_0", false, [(0, ScaleKind.Half)]),
        (GgufType.Q5_1, "dequantize_row_q5_1", true, [(0, ScaleKind.Half), (2, ScaleKind.Half)]),
        (GgufType.Q8_0, "dequantize_row_q8_0", false, [(0, ScaleKind.Half)]),
        (GgufType.Q2_K, "dequantize_row_q2_K", true, [(80, ScaleKind.Half), (82, ScaleKind.Half)]),
        (GgufType.Q3_K, "dequantize_row_q3_K", false, [(108, ScaleKind.Half)]),
        (GgufType.Q4_K, "dequantize_row_q4_K", true, [(0, ScaleKind.Half), (2, ScaleKind.Half)]),
        (GgufType.Q5_K, "dequantize_row_q5_K", true, [(0, ScaleKind.Half), (2, ScaleKind.Half)]),
        (GgufType.Q6_K, "dequantize_row_q6_K", false, [(208, ScaleKind.Half)]),
        (GgufType.Q8_K, "dequantize_row_q8_K", false, [(0, ScaleKind.Single)]),
        (GgufType.IQ4_NL, "dequantize_row_iq4_nl", false, [(0, ScaleKind.Half)]),
        (GgufType.IQ4_XS, "dequantize_row_iq4_xs", false, [(0, ScaleKind.Half)]),
        (GgufType.TQ1_0, "dequantize_row_tq1_0", false, [(52, ScaleKind.Half)]),
        (GgufType.TQ2_0, "dequantize_row_tq2_0", false, [(64, ScaleKind.Half)]),
        (GgufType.MXFP4, "dequantize_row_mxfp4", false, [(0, ScaleKind.E8M0)]),
    ];

    private static byte[] RandomBlocks(ISourceTensorDecoder decoder, (int Offset, ScaleKind Kind)[] scales, int seed)
    {
        var random = new Random(seed);
        var bytes = new byte[Blocks * decoder.BlockBytes];
        random.NextBytes(bytes);
        for (var block = 0; block < Blocks; block++)
        {
            foreach (var (offset, kind) in scales)
            {
                var field = bytes.AsSpan((block * decoder.BlockBytes) + offset);
                var value = (random.NextDouble() * 3.5) - 1.75;
                switch (kind)
                {
                    case ScaleKind.Half:
                        BinaryPrimitives.WriteHalfLittleEndian(field, (Half)value);
                        break;
                    case ScaleKind.Single:
                        BinaryPrimitives.WriteSingleLittleEndian(field, (float)value);
                        break;
                    case ScaleKind.E8M0:
                        field[0] = (byte)random.Next(100, 151);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(scales));
                }
            }
        }

        return bytes;
    }

    private static float Difference(float actual, float expected) =>
        actual == expected ? 0f : MathF.Abs(actual - expected) is var difference && float.IsNaN(difference) ? float.PositiveInfinity : difference;

    private enum ScaleKind
    {
        Half,
        Single,
        E8M0,
    }

    private sealed class GgmlLibrary
    {
        private readonly nint _handle;

        private GgmlLibrary(nint handle)
        {
            _handle = handle;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void RowDecoder(nint source, nint destination, long count);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate long TypeQuery(int type);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint TypeSize(int type);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nint TypeNameQuery(int type);

        // ggml installs a process-wide terminate handler when it loads, so it is loaded once and never freed.
        public static GgmlLibrary Instance => Shared.Value;

        private static readonly Lazy<GgmlLibrary> Shared = new(Load);

        private static GgmlLibrary Load()
        {
            var fileName = OperatingSystem.IsWindows() ? "ggml-base.dll"
                : OperatingSystem.IsMacOS() ? "libggml-base.dylib"
                : "libggml-base.so";
            var native = Path.Combine(AppContext.BaseDirectory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native");
            var path = new[] { Path.Combine(native, fileName), Path.Combine(native, "noavx", fileName) }.FirstOrDefault(File.Exists);
            return path is not null
                ? new GgmlLibrary(NativeLibrary.Load(path))
                : throw new InvalidOperationException($"The ggml reference library is missing under {native}; the oracle test does not skip.");
        }

        public long BlockElements(GgufType type) => Get<TypeQuery>("ggml_blck_size")((int)type);

        public long BlockBytes(GgufType type) => (long)Get<TypeSize>("ggml_type_size")((int)type);

        public string TypeName(GgufType type) => Marshal.PtrToStringUTF8(Get<TypeNameQuery>("ggml_type_name")((int)type)) ?? string.Empty;

        public float[] Dequantize(string function, byte[] source, int elements)
        {
            var output = new float[elements];
            var sourceHandle = GCHandle.Alloc(source, GCHandleType.Pinned);
            var outputHandle = GCHandle.Alloc(output, GCHandleType.Pinned);
            try
            {
                Get<RowDecoder>(function)(sourceHandle.AddrOfPinnedObject(), outputHandle.AddrOfPinnedObject(), elements);
            }
            finally
            {
                sourceHandle.Free();
                outputHandle.Free();
            }

            return output;
        }

        private T Get<T>(string name)
            where T : Delegate => Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(_handle, name));
    }
}
