using System.Buffers.Binary;
using System.Text;
using ManagedCode.Synapse.Cli.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;
using ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

[NotInParallel]
public sealed class ModelConverterTests
{
    [Test]
    public async Task ConvertsSafeTensorsExplicitlyToNativePackage()
    {
        using var fixture = new NativeConversionFixture();
        var info = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination,
            new ModelConversionOptions(fixture.Graph));
        await Assert.That(info.SourceFormat).IsEqualTo("safetensors");
        await Assert.That(info.Sources.Length).IsEqualTo(2);
        await Assert.That(File.Exists(fixture.Destination)).IsTrue();
    }

    [Test]
    public async Task ConvertCliAcceptsExplicitSafeTensorsGraph()
    {
        using var fixture = new NativeConversionFixture();
        var result = await ModelCommand.RunAsync(["convert", "--source", fixture.Source,
            "--graph", fixture.Graph, "--output", fixture.Destination]);
        await Assert.That(result).IsEqualTo(0);
    }

    [Test]
    public async Task GgufConversionPreservesActualProvenanceAndVersionOneInspection()
    {
        using var fixture = new CompiledPackageFixture();
        var length = new FileInfo(fixture.Source).Length;
        var info = await ModelConverter.ConvertAsync(fixture.Source, fixture.Destination);
        File.Delete(fixture.Source);
        var inspected = ModelConverter.Inspect(fixture.Destination);
        await Assert.That(info.SourceFormat).IsEqualTo("gguf");
        await Assert.That(inspected.Sources[0].Length).IsEqualTo(length);
        await Assert.That(inspected.Identity).IsEqualTo(info.Identity);
    }

    [Test]
    public async Task OnnxConversionPublishesAnExecutableIndependentNativeGraph()
    {
        using var fixture = new OnnxFixture();
        var source = fixture.Write(OnnxFixture.Model(OnnxFixture.Graph(
            [OnnxFixture.Node("MatMul", ["x", "w"])],
            [OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4])],
            [OnnxFixture.Value("x", "batch", 2)], [OnnxFixture.Value("y", "batch", 2)])));
        var destination = Path.ChangeExtension(source, ".synapse");
        var info = await ModelConverter.ConvertAsync(source, destination,
            new(Dimensions: new Dictionary<string, ConversionBound> { ["batch"] = new(1, 4) }));
        File.Delete(source);
        var output = NativeGraphPackage.Execute(destination, new Dictionary<string, float[]> { ["x"] = [1, 2, 3, 4] },
            new Dictionary<string, long[]> { ["x"] = [2, 2] });
        await Assert.That(info.SourceFormat).IsEqualTo("onnx");
        await Assert.That(output["y"]).IsEquivalentTo([7f, 10f, 15f, 22f]);
    }

    [Test]
    public async Task OnnxConversionRejectsUnusedDimensionBounds()
    {
        using var fixture = new OnnxFixture();
        var source = fixture.Write(OnnxFixture.Model(OnnxFixture.Graph(
            [OnnxFixture.Node("Identity", ["x"])], [], [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y", 2)])));
        var destination = Path.ChangeExtension(source, ".synapse");
        await Assert.That(async () => await ModelConverter.ConvertAsync(source, destination,
            new(Dimensions: new Dictionary<string, ConversionBound> { ["typo"] = new(1, 4) })))
            .Throws<InvalidDataException>();
        await Assert.That(File.Exists(destination)).IsFalse();
    }
}

internal sealed class NativeConversionFixture : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "synapse-conversion-" + Guid.NewGuid().ToString("N"));

    internal NativeConversionFixture()
    {
        _ = Directory.CreateDirectory(_directory);
        var header = Encoding.UTF8.GetBytes(/*lang=json,strict*/ "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}");
        using var file = File.Create(Source);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(length, header.Length);
        file.Write(length);
        file.Write(header);
        using var writer = new BinaryWriter(file);
        foreach (var value in new[] { 1f, 2f, 3f, 4f })
        {
            writer.Write(value);
        }

        File.WriteAllText(Graph, /*lang=json,strict*/ """
            {"schema_version":1,"inputs":[{"name":"x","shape":[{"symbol":"batch","minimum":1,"maximum":4},{"symbol":null,"minimum":2,"maximum":2}]}],
             "outputs":["y"],"nodes":[{"name":"project","operation":"Linear","inputs":["x","w"],"output":"y"}]}
            """);
    }

    internal string Source => Path.Combine(_directory, "weights.safetensors");
    internal string Graph => Path.Combine(_directory, "graph.json");
    internal string Destination => Path.Combine(_directory, "model.synapse");
    public void Dispose() => Directory.Delete(_directory, recursive: true);
}
