using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.Catalog;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class ModelPackageDownloaderTests
{
    [Test]
    public async Task OversizedPayloadStopsBeforeWritingBeyondDeclaredSize()
    {
        var temporary = Path.Combine(Path.GetTempPath(), $"synapse-download-{Guid.NewGuid():N}.partial");
        var file = new ModelPackageFile(
            "model.bin",
            new Uri("https://huggingface.co/example/model.bin"),
            1,
            new string('0', 64));
        await using var source = new MemoryStream([0x01, 0x02]);
        try
        {
            await Assert.That(async () => await ModelPackageDownloader.CopyPayloadAsync(
                source,
                temporary,
                file,
                CancellationToken.None)).Throws<InvalidDataException>();
            await Assert.That(new FileInfo(temporary).Length).IsLessThanOrEqualTo(file.SizeBytes);
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    [Test]
    public async Task RedirectTrustRejectsHttpAndUnrelatedHosts()
    {
        await Assert.That(() => ModelPackageDownloader.EnsureTrustedUri(
            new Uri("http://huggingface.co/model.bin"),
            "huggingface.co",
            isInitial: true)).Throws<InvalidDataException>();
        await Assert.That(() => ModelPackageDownloader.EnsureTrustedUri(
            new Uri("https://example.test/model.bin"),
            "huggingface.co",
            isInitial: false)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task RedirectTrustAcceptsHuggingFaceStorageHosts()
    {
        var uri = new Uri("https://cas-bridge.xethub.hf.co/model.bin");
        ModelPackageDownloader.EnsureTrustedUri(
            uri,
            "huggingface.co",
            isInitial: false);
        await Assert.That(uri.Scheme).IsEqualTo(Uri.UriSchemeHttps);
    }
}
