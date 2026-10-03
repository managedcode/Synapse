using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

[NotInParallel]
public sealed class NativeGraphPackageTests
{
    [Test]
    public async Task PreparedPackageExecutesTwoBatchSizesWithoutSource()
    {
        using var fixture = new NativeConversionFixture();
        _ = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph));
        File.Delete(fixture.Source);
        File.Delete(fixture.Graph);
        var one = NativeGraphPackage.Execute(fixture.Destination, new Dictionary<string, float[]> { ["x"] = [1, 2] },
            new Dictionary<string, long[]> { ["x"] = [1, 2] });
        var two = NativeGraphPackage.Execute(fixture.Destination, new Dictionary<string, float[]> { ["x"] = [1, 2, 3, 4] },
            new Dictionary<string, long[]> { ["x"] = [2, 2] });
        await Assert.That(one["y"]).IsEquivalentTo([5f, 11f]);
        await Assert.That(two["y"]).IsEquivalentTo([5f, 11f, 11f, 25f]);
        await Assert.That(ModelConverter.Inspect(fixture.Destination).TensorCount).IsEqualTo(1);
    }

    [Test]
    public async Task ConversionIsDeterministicAndPreservesDestinationOnFailure()
    {
        using var fixture = new NativeConversionFixture();
        _ = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph));
        var second = fixture.Destination + ".second.synapse";
        try
        {
            _ = await ModelConverter.ConvertAsync(fixture.Source, second, new(fixture.Graph));
            var expected = File.ReadAllBytes(fixture.Destination);
            await Assert.That(File.ReadAllBytes(second)).IsEquivalentTo(expected);
            await Assert.That(async () => await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph)))
                .Throws<IOException>();
            await Assert.That(File.ReadAllBytes(fixture.Destination)).IsEquivalentTo(expected);
        }
        finally
        {
            File.Delete(second);
        }
    }

    [Test]
    public async Task RejectsPayloadCorruptionAndAdversarialTensorBounds()
    {
        using var fixture = new NativeConversionFixture();
        _ = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph));
        var bytes = File.ReadAllBytes(fixture.Destination);
        var offset = (int)BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(16));
        bytes[offset] ^= 1;
        File.WriteAllBytes(fixture.Destination, bytes);
        await Assert.That(() => NativeGraphPackage.Inspect(fixture.Destination)).Throws<InvalidDataException>();
        bytes[offset] ^= 1;
        var length = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(12));
        var manifest = Encoding.UTF8.GetString(bytes, 96, length).Replace("\"offset\":0", "\"offset\":1", StringComparison.Ordinal);
        var encoded = Encoding.UTF8.GetBytes(manifest);
        encoded.CopyTo(bytes, 96);
        SHA256.HashData(encoded).CopyTo(bytes, 32);
        File.WriteAllBytes(fixture.Destination, bytes);
        await Assert.That(() => NativeGraphPackage.Inspect(fixture.Destination)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task CancellationAndUnsupportedSourcesPublishNothing()
    {
        using var fixture = new NativeConversionFixture();
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        await Assert.That(async () => await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph), cancelled.Token))
            .Throws<OperationCanceledException>();
        await Assert.That(async () => await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination))
            .Throws<NotSupportedException>();
        await Assert.That(File.Exists(fixture.Destination)).IsFalse();
        await Assert.That(Directory.EnumerateFiles(Path.GetDirectoryName(fixture.Destination)!, "*.partial.synapse").Any()).IsFalse();
    }

    [Test]
    public async Task RunCliRejectsDuplicateInputNames()
    {
        using var fixture = new NativeConversionFixture();
        _ = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph));
        var inputs = Path.ChangeExtension(fixture.Source, ".inputs.json");
        File.WriteAllText(inputs, /*lang=json,strict*/ "{\"x\":{\"shape\":[1,2],\"data\":[1,2]},\"x\":{\"shape\":[1,2],\"data\":[3,4]}}");
        var result = await ModelCommand.RunAsync(["run", "--model", fixture.Destination, "--inputs", inputs]);
        await Assert.That(result).IsEqualTo(1);
    }

    [Test]
    public async Task ReaderRejectsDuplicateManifestPropertiesEvenWithValidDigests()
    {
        using var fixture = new NativeConversionFixture();
        _ = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination, new(fixture.Graph));
        var bytes = File.ReadAllBytes(fixture.Destination);
        var oldOffset = (int)BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(16));
        var length = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(12));
        var json = Encoding.UTF8.GetString(bytes, 96, length).Replace("\"schema_version\":2,",
            "\"schema_version\":2,\"schema_version\":2,", StringComparison.Ordinal);
        var manifest = Encoding.UTF8.GetBytes(json);
        var offset = (96 + manifest.Length + 63) / 64 * 64;
        var changed = new byte[offset + bytes.Length - oldOffset];
        bytes.AsSpan(0, 96).CopyTo(changed);
        manifest.CopyTo(changed, 96);
        bytes.AsSpan(oldOffset).CopyTo(changed.AsSpan(offset));
        BinaryPrimitives.WriteInt32LittleEndian(changed.AsSpan(12), manifest.Length);
        BinaryPrimitives.WriteInt64LittleEndian(changed.AsSpan(16), offset);
        BinaryPrimitives.WriteInt64LittleEndian(changed.AsSpan(24), changed.Length);
        SHA256.HashData(manifest).CopyTo(changed, 32);
        File.WriteAllBytes(fixture.Destination, changed);
        await Assert.That(() => NativeGraphPackage.Inspect(fixture.Destination)).Throws<InvalidDataException>();
    }
}
