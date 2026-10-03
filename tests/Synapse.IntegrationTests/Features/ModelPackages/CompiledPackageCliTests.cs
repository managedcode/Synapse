using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

[NotInParallel]
public sealed class CompiledPackageCliTests
{
    [Test]
    public async Task CompileCommandPublishesExecutablePackage()
    {
        var source = TinyQwen2Gguf.Write(new(1, 1, 1, 64, 256, 64, 32), seed: 19);
        var destination = Path.ChangeExtension(source, ".synapse");
        try
        {
            var result = await ModelCommand.RunAsync(["compile", "--source", source, "--output", destination]);
            await Assert.That(result).IsEqualTo(0);
            await Assert.That(File.Exists(destination)).IsTrue();
        }
        finally
        {
            File.Delete(source);
            File.Delete(destination);
        }
    }
}
