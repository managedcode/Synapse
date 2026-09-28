using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelLoading;

public sealed class GgufStartupTests
{
    [Test]
    public async Task OpeningSkipsUnusedMetadataStringArraysWithoutAllocating()
    {
        using (GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath()))
        {
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        using var file = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        await Assert.That(file.GetRequiredString("general.architecture")).IsEqualTo("qwen2");
        await Assert.That(file.Metadata.ContainsKey("tokenizer.ggml.tokens")).IsFalse();
        await Assert.That(file.Tensors.Count).IsEqualTo(291);
        await Assert.That(allocated).IsLessThan(2L * 1024 * 1024);
    }
}

public sealed class GgufPrefetchRangeTests
{
    [Test]
    public async Task PrefetchRangesCoverDenseTensorsAndSkipTheSparseEmbedding()
    {
        using var file = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        var embedding = file.GetRequiredTensor("token_embd.weight");
        var ranges = file.GetTensorDataRanges(["token_embd.weight"]);
        var (start, _) = file.GetTensorDataRanges([])[0];
        var baseAddress = start - file.Tensors.Values.Min(tensor => tensor.Offset);
        var fileRanges = ranges.Select(range => (Offset: range.Start - baseAddress, range.Length)).ToArray();

        await Assert.That(ranges.Count).IsEqualTo(2);
        await Assert.That(file.Tensors.Values.Where(tensor => tensor.Name != embedding.Name)
            .All(tensor => fileRanges.Any(range =>
                range.Offset <= tensor.Offset && tensor.Offset + tensor.ByteLength <= range.Offset + range.Length)))
            .IsTrue();
        await Assert.That(fileRanges.Any(range =>
            range.Offset < embedding.Offset + embedding.ByteLength &&
            embedding.Offset < range.Offset + range.Length)).IsFalse();
    }
}
