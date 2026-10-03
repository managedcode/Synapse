using System.Buffers.Binary;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class ScalarSourceAllocationTests
{
    // TEST-PKG-003-4: model conversion must not create garbage per scalar.
    [Test]
    public async Task ScalarDecodingAllocatesNoMemoryPerElement()
    {
        ISourceTensorDecoder[] decoders =
        [
            SourceEncodings.Fp32, SourceEncodings.Fp64, SourceEncodings.Fp16,
            SourceEncodings.Bf16, SourceEncodings.Fp8E4M3Fn, SourceEncodings.Fp8E5M2,
        ];
        foreach (var decoder in decoders)
        {
            var source = new byte[checked((int)decoder.GetByteLength(4096))];
            var destination = new float[4096];
            for (var warmup = 0; warmup < 8; warmup++)
            {
                decoder.Decode(source, destination);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();
            decoder.Decode(source, destination);
            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            await Assert.That(allocated).IsEqualTo(0L).Because(decoder.EncodingId);
            await Assert.That(destination.All(value => value == 0f)).IsTrue();
        }
    }

    [Test]
    public async Task LateNonFiniteScalarLeavesWholeDestinationUnchanged()
    {
        var source = new byte[4096 * sizeof(double)];
        BinaryPrimitives.WriteDoubleLittleEndian(source.AsSpan(source.Length - sizeof(double)), double.MaxValue);
        var destination = Enumerable.Repeat(42f, 4096).ToArray();

        await Assert.That(() => SourceEncodings.Fp64.Decode(source, destination))
            .Throws<SourceEncodingException>();
        await Assert.That(destination.All(value => value == 42f)).IsTrue();
    }
}
