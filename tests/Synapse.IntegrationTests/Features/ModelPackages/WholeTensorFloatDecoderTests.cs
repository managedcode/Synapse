using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class WholeTensorFloatDecoderTests
{
    // TEST-PKG-003-5: faster source preparation must preserve every stored finite bit pattern.
    [Test]
    public async Task Fp32WholeTensorPreservesFiniteBitsIncludingSignedZeroAndSubnormals()
    {
        uint[] anchors = [0, 0x8000_0000, 1, 0x8000_0001, 0x007F_FFFF, 0x0080_0000, 0x3F80_0000, 0xBF80_0000, 0x7F7F_FFFF, 0xFF7F_FFFF];
        var random = new Random(1703);
        var bits = anchors.Concat(Enumerable.Range(0, 4096).Select(_ => (uint)random.NextInt64(0, 1L << 32)))
            .Where(value => (value & 0x7F80_0000) != 0x7F80_0000).ToArray();
        var source = new byte[bits.Length * sizeof(float)];
        for (var index = 0; index < bits.Length; index++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(source.AsSpan(index * sizeof(float)), bits[index]);
        }

        var destination = new float[bits.Length];
        SourceEncodings.Fp32.Decode(source, destination);

        await Assert.That(destination.Select(BitConverter.SingleToUInt32Bits).SequenceEqual(bits)).IsTrue();
    }

    [Test]
    [Arguments("BF16")]
    [Arguments("F16")]
    public async Task EveryFiniteSixteenBitCodeDecodesExactly(string dtype)
    {
        _ = SourceEncodings.TryGetSafeTensors(dtype, out var decoder);
        var mask = dtype == "BF16" ? 0x7F80 : 0x7C00;
        var codes = Enumerable.Range(0, 1 << 16).Where(code => (code & mask) != mask).Select(code => (ushort)code).ToArray();
        var source = new byte[codes.Length * sizeof(ushort)];
        for (var index = 0; index < codes.Length; index++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(source.AsSpan(index * sizeof(ushort)), codes[index]);
        }

        var destination = new float[codes.Length];
        decoder!.Decode(source, destination);

        var expected = codes.Select(code => dtype == "BF16" ? (uint)code << 16 :
            BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(code)));
        await Assert.That(destination.Select(BitConverter.SingleToUInt32Bits).SequenceEqual(expected)).IsTrue();
    }

    [Test]
    [Arguments("F32", 0x7F80_0000U)]
    [Arguments("F32", 0xFF80_0000U)]
    [Arguments("F32", 0x7FC0_0001U)]
    [Arguments("BF16", 0x7F80U)]
    [Arguments("BF16", 0xFF80U)]
    [Arguments("BF16", 0x7FC1U)]
    [Arguments("F16", 0x7C00U)]
    [Arguments("F16", 0xFC00U)]
    [Arguments("F16", 0x7E01U)]
    public async Task LateInvalidFloatNamesItsIndexAndLeavesAllOutputUnchanged(string dtype, uint invalid)
    {
        _ = SourceEncodings.TryGetSafeTensors(dtype, out var decoder);
        const int Count = 4096;
        var source = new byte[checked((int)decoder!.GetByteLength(Count))];
        if (decoder.BlockBytes == sizeof(float))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(source.AsSpan()[^sizeof(float)..], invalid);
        }
        else
        {
            BinaryPrimitives.WriteUInt16LittleEndian(source.AsSpan()[^sizeof(ushort)..], checked((ushort)invalid));
        }

        var destination = Enumerable.Repeat(42f, Count).ToArray();
        var failure = await Assert.That(() => decoder.Decode(source, destination)).Throws<SourceEncodingException>();

        await Assert.That(failure!.Failure).IsEqualTo(SourceEncodingFailure.NonFiniteValue);
        await Assert.That(failure.Message).Contains("block 4095");
        await Assert.That(destination.All(value => value == 42f)).IsTrue();
    }

    [Test]
    [Arguments("F32")]
    [Arguments("BF16")]
    [Arguments("F16")]
    public async Task WholeTensorLengthsAndOverflowRetainTheSourceContract(string dtype)
    {
        _ = SourceEncodings.TryGetSafeTensors(dtype, out var decoder);
        var destination = new[] { 42f, 42f };

        await Assert.That(() => decoder!.GetByteLength(0)).Throws<ArgumentException>();
        await Assert.That(() => decoder!.GetByteLength(-1)).Throws<ArgumentException>();
        await Assert.That(() => decoder!.GetByteLength(long.MaxValue)).Throws<OverflowException>();
        await Assert.That(() => decoder!.Decode(new byte[decoder.BlockBytes], destination)).Throws<ArgumentException>();
        await Assert.That(() => decoder!.Decode([], [])).Throws<ArgumentException>();
        await Assert.That(destination.All(value => value == 42f)).IsTrue();
    }

    [Test]
    [Arguments("F32")]
    [Arguments("BF16")]
    [Arguments("F16")]
    public async Task FirstInvalidAtVectorBoundariesAndTailPreservesEveryDestinationValue(string dtype)
    {
        _ = SourceEncodings.TryGetSafeTensors(dtype, out var decoder);
        var width = dtype == "F32" ? Vector<uint>.Count : Vector<ushort>.Count;
        var count = (3 * width) + 3;
        foreach (var invalidIndex in new[] { 0, width - 1, width, width + 1, 3 * width, count - 1 })
        {
            var source = new byte[checked((int)decoder!.GetByteLength(count))];
            WriteNonFinite(source, dtype, invalidIndex);
            WriteNonFinite(source, dtype, count - 1);
            var destination = Enumerable.Repeat(42f, count).ToArray();

            var failure = await Assert.That(() => decoder.Decode(source, destination)).Throws<SourceEncodingException>();

            await Assert.That(failure!.Failure).IsEqualTo(SourceEncodingFailure.NonFiniteValue);
            await Assert.That(failure.Message).Contains($"block {invalidIndex} ");
            await Assert.That(destination.All(value => value == 42f)).IsTrue();
        }
    }

    [Test]
    public async Task OverlappingFp32RetainsExistingForwardCopySemantics()
    {
        float[] storage = [1, 2, 3, 4];
        SourceEncodings.Fp32.Decode(MemoryMarshal.AsBytes(storage.AsSpan(0, 3)), storage.AsSpan(1, 3));

        await Assert.That(storage.SequenceEqual([1f, 1f, 1f, 1f])).IsTrue();
    }

    [Test]
    public async Task OverlappingBf16RetainsExistingForwardExpansionSemantics()
    {
        var storage = new float[3];
        var bytes = MemoryMarshal.AsBytes(storage.AsSpan());
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, 0x3F80);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes[2..], 0x4000);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes[4..], 0x4040);
        SourceEncodings.Bf16.Decode(bytes[..6], storage);

        // Forward writes can replace source codes that have not yet been read.
        await Assert.That(storage.Select(BitConverter.SingleToUInt32Bits).SequenceEqual([0x3F80_0000U, 0x3F80_0000U, 0U])).IsTrue();
    }

    [Test]
    public async Task OverlappingFp16RetainsExistingForwardExpansionSemantics()
    {
        var storage = new float[3];
        var bytes = MemoryMarshal.AsBytes(storage.AsSpan());
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, 0x3C00);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes[2..], 0x4000);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes[4..], 0x4200);
        SourceEncodings.Fp16.Decode(bytes[..6], storage);

        await Assert.That(storage.SequenceEqual([1f, 1.875f, 0f])).IsTrue();
    }

    private static void WriteNonFinite(byte[] source, string dtype, int index)
    {
        if (dtype == "F32")
        {
            BinaryPrimitives.WriteUInt32LittleEndian(source.AsSpan(index * sizeof(float)), 0x7FC0_0001);
        }
        else
        {
            BinaryPrimitives.WriteUInt16LittleEndian(source.AsSpan(index * sizeof(ushort)), dtype == "BF16" ? (ushort)0x7FC1 : (ushort)0x7E01);
        }
    }
}
