using System.Buffers.Binary;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class SourceTensorDecoderTests
{
    [Test]
    public async Task GgufQ40GoldenBlockDecodes()
    {
        var block = new byte[18];
        BinaryPrimitives.WriteUInt16LittleEndian(block, 0x3C00);
        block.AsSpan(2).Fill(0x88);
        block[2] = 0x9F;
        block[3] = 0x08;

        var values = Decode(GgufType.Q4_0, block, 32);

        var expected = new float[32];
        expected[0] = 7f;
        expected[16] = 1f;
        expected[17] = -8f;
        await Assert.That(values).IsEquivalentTo(expected);
    }

    [Test]
    public async Task GgufQ50GoldenBlockUsesHighBits()
    {
        var block = new byte[22];
        BinaryPrimitives.WriteUInt16LittleEndian(block, 0x3800);
        BinaryPrimitives.WriteUInt32LittleEndian(block.AsSpan(2), 0x0001_0001);
        block[6] = 0x0F;

        var values = Decode(GgufType.Q5_0, block, 32);

        var expected = Enumerable.Repeat(-8f, 32).ToArray();
        expected[0] = 7.5f;
        expected[16] = 0f;
        await Assert.That(values).IsEquivalentTo(expected);
    }

    [Test]
    public async Task GgufQ80GoldenBlockDecodes()
    {
        var block = new byte[34];
        BinaryPrimitives.WriteUInt16LittleEndian(block, 0x3400);
        block[2] = 0x7F;
        block[3] = 0x81;
        block[4] = 0x80;

        var values = Decode(GgufType.Q8_0, block, 32);

        await Assert.That(values.Take(4)).IsEquivalentTo([31.75f, -31.75f, -32f, 0f]);
    }

    [Test]
    public async Task FloatFormatsDecodeExactly()
    {
        var half = new byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(half, 0x3C00);
        BinaryPrimitives.WriteUInt16LittleEndian(half.AsSpan(2), 0xC500);
        var brain = new byte[4];
        BinaryPrimitives.WriteUInt16LittleEndian(brain, 0x3F80);
        BinaryPrimitives.WriteUInt16LittleEndian(brain.AsSpan(2), 0xC040);

        await Assert.That(Decode(SourceEncodings.Fp16, half, 2)).IsEquivalentTo([1f, -5f]);
        await Assert.That(Decode(SourceEncodings.Bf16, brain, 2)).IsEquivalentTo([1f, -3f]);
    }

    [Test]
    public async Task Fp8E5M2MatchesTopByteOfHalfForEveryFiniteCode()
    {
        var codes = Enumerable.Range(0, 256)
            .Select(code => (byte)code)
            .Where(code => (code & 0x7C) != 0x7C)
            .ToArray();

        var values = Decode(SourceEncodings.Fp8E5M2, codes, codes.Length);

        for (var index = 0; index < codes.Length; index++)
        {
            var expected = (float)BitConverter.UInt16BitsToHalf((ushort)(codes[index] << 8));
            await Assert.That(BitConverter.SingleToInt32Bits(values[index])).IsEqualTo(BitConverter.SingleToInt32Bits(expected));
        }
    }

    [Test]
    public async Task Fp8E4M3FnAnchorsDecode()
    {
        byte[] codes = [0x38, 0x7E, 0x01, 0x08, 0xB8, 0x00, 0xFE];

        var values = Decode(SourceEncodings.Fp8E4M3Fn, codes, codes.Length);

        await Assert.That(values).IsEquivalentTo([1f, 448f, MathF.ScaleB(1f, -9), MathF.ScaleB(1f, -6), -1f, 0f, -448f]);
    }

    [Test]
    public async Task NonFiniteValuesAndScalesRejectedBeforeWriting()
    {
        var destination = Enumerable.Repeat(42f, 32).ToArray();
        var badBlock = new byte[18];
        BinaryPrimitives.WriteUInt16LittleEndian(badBlock, 0x7E00);
        byte[] fp8NaN = [0x7F];

        var scaleFailure = await Assert.That(() => SourceEncodings.GetGguf((uint)GgufType.Q4_0).Decode(badBlock, destination))
            .Throws<SourceEncodingException>();
        var valueFailure = await Assert.That(() => SourceEncodings.Fp8E4M3Fn.Decode(fp8NaN, new float[1]))
            .Throws<SourceEncodingException>();

        await Assert.That(scaleFailure!.Failure).IsEqualTo(SourceEncodingFailure.InvalidScale);
        await Assert.That(valueFailure!.Failure).IsEqualTo(SourceEncodingFailure.NonFiniteValue);
        await Assert.That(destination.All(value => value == 42f)).IsTrue();
    }

    [Test]
    public async Task GgufRegistryNamesSupportedAndUnsupportedTypes()
    {
        (uint Type, string Name, bool Supported)[] expectations =
        [
            (0, "f32", true), (1, "f16", true), (2, "q4_0", true), (3, "q4_1", true), (6, "q5_0", true),
            (7, "q5_1", true), (8, "q8_0", true), (9, "q8_1", false), (10, "q2_K", true), (11, "q3_K", true),
            (12, "q4_K", true), (13, "q5_K", true), (14, "q6_K", true), (15, "q8_K", true), (16, "iq2_xxs", false),
            (17, "iq2_xs", false), (18, "iq3_xxs", false), (19, "iq1_s", false), (20, "iq4_nl", true), (21, "iq3_s", false),
            (22, "iq2_s", false), (23, "iq4_xs", true), (28, "f64", true), (29, "iq1_m", false), (30, "bf16", true),
            (34, "tq1_0", true), (35, "tq2_0", true), (39, "mxfp4", true), (40, "nvfp4", false), (41, "q1_0", false),
        ];

        foreach (var (type, name, supported) in expectations)
        {
            await Assert.That(SourceEncodings.GetGgufTypeName(type)).IsEqualTo(name);
            await Assert.That(SourceEncodings.TryGetGguf(type, out _)).IsEqualTo(supported);
        }

        var failure = await Assert.That(() => SourceEncodings.GetGguf(16)).Throws<SourceEncodingException>();
        await Assert.That(failure!.Failure).IsEqualTo(SourceEncodingFailure.UnsupportedType);
        await Assert.That(SourceEncodings.TryGetGguf(4, out _)).IsFalse();
        await Assert.That(SourceEncodings.TryGetGguf(99, out _)).IsFalse();
    }

    [Test]
    public async Task SafeTensorsDtypesResolve()
    {
        foreach (var dtype in new[] { "F32", "F16", "BF16", "F64", "F8_E4M3", "F8_E5M2" })
        {
            await Assert.That(SourceEncodings.TryGetSafeTensors(dtype, out _)).IsTrue();
        }

        await Assert.That(SourceEncodings.TryGetSafeTensors("I8", out _)).IsFalse();
    }

    [Test]
    public async Task BlockLengthsAreExact()
    {
        var q4k = SourceEncodings.GetGguf((uint)GgufType.Q4_K);

        await Assert.That(q4k.GetByteLength(512)).IsEqualTo(288L);
        await Assert.That(() => q4k.GetByteLength(300)).Throws<ArgumentException>();
        await Assert.That(() => q4k.Decode(new byte[144], new float[512])).Throws<ArgumentException>();
    }

    private static float[] Decode(GgufType type, byte[] source, int elements) =>
        Decode(SourceEncodings.GetGguf((uint)type), source, elements);

    private static float[] Decode(ISourceTensorDecoder decoder, byte[] source, int elements)
    {
        var values = new float[elements];
        decoder.Decode(source, values);
        return values;
    }
}
