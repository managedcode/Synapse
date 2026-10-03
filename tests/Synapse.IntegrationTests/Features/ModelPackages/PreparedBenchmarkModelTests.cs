using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class PreparedBenchmarkModelTests
{
    // TEST-PKG-004-10: benchmarking consumes an explicitly prepared artifact without hidden conversion.
    [Test]
    public async Task BenchmarkRefusesUnpreparedSourceWithoutCreatingOutput()
    {
        var source = TinyQwen2Gguf.Write(Shape(), seed: 1701);
        var prepared = Path.ChangeExtension(source, ".synapse");
        try
        {
            await Assert.That(() => PreparedBenchmarkModel.Require(source)).Throws<FileNotFoundException>();
            await Assert.That(File.Exists(prepared)).IsFalse();
        }
        finally
        {
            File.Delete(source);
            File.Delete(prepared);
        }
    }

    [Test]
    public async Task BenchmarkVerifiesPreparedArtifactAgainstSourceIdentity()
    {
        var source = TinyQwen2Gguf.Write(Shape(), seed: 1702);
        var prepared = Path.ChangeExtension(source, ".synapse");
        try
        {
            _ = await CompiledPackageCompiler.CompileAsync(source, prepared);
            await using (var stream = File.OpenWrite(source))
            {
                stream.Position = stream.Length - 1;
                stream.WriteByte(0xFF);
            }

            await Assert.That(() => PreparedBenchmarkModel.Require(source)).Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(source);
            File.Delete(prepared);
        }
    }

    [Test]
    [Arguments(false)]
    [Arguments(true)]
    public async Task BenchmarkRevalidatesChangedFilesAfterSuccessfulPreparation(bool changePackage)
    {
        var source = TinyQwen2Gguf.Write(Shape(), seed: 1703);
        var prepared = Path.ChangeExtension(source, ".synapse");
        try
        {
            _ = await CompiledPackageCompiler.CompileAsync(source, prepared);
            await Assert.That(PreparedBenchmarkModel.Require(source)).IsEqualTo(prepared);
            var changed = changePackage ? prepared : source;
            var timestamp = File.GetLastWriteTimeUtc(changed);
            var bytes = await File.ReadAllBytesAsync(changed);
            bytes[^1] ^= 1;
            await File.WriteAllBytesAsync(changed, bytes);
            File.SetLastWriteTimeUtc(changed, timestamp);

            await Assert.That(() => PreparedBenchmarkModel.Require(source)).Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(source);
            File.Delete(prepared);
        }
    }

    private static TinyQwen2Gguf.Shape Shape() => new(
        Layers: 1, Heads: 2, KeyValueHeads: 1, HeadDimension: 32,
        FeedForward: 128, Vocabulary: 323, Context: 64);
}
