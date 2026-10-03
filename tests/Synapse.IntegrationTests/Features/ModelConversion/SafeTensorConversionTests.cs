using System.Text;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

/// <summary>TEST-CNV-001-1 and TEST-CNV-001-4: real SafeTensors import and rejection.</summary>
public sealed class SafeTensorConversionTests
{
    [Test]
    public async Task ImportNormalizesFp32Fp16AndBf16Exactly()
    {
        using var fixture = new SafeTensorConversionFixture();
        fixture.Write(
            new("weight", "F32", [1, 2], 4, SafeTensorConversionFixture.Singles(1.25f, -2)),
            new("half", "F16", [2], 2, SafeTensorConversionFixture.Halves(0x0001, 0xBC00)),
            new("brain", "BF16", [2], 2, SafeTensorConversionFixture.Halves(0x0001, 0xC000)));
        var model = Import(fixture);
        await Assert.That(model.SourceFormat).IsEqualTo("safetensors");
        await Assert.That(model.Graph.Nodes[0].Operation).IsEqualTo("Linear");
        await Assert.That(model.Tensors.Single(tensor => tensor.Name == "weight").Data).IsEquivalentTo([1.25f, -2f]);
        await Assert.That(model.Tensors.Single(tensor => tensor.Name == "half").Data).IsEquivalentTo([MathF.ScaleB(1, -24), -1f]);
        await Assert.That(model.Tensors.Single(tensor => tensor.Name == "brain").Data).IsEquivalentTo([BitConverter.UInt32BitsToSingle(0x00010000), -2f]);
        await Assert.That(model.Tensors[0].Shape).IsEquivalentTo([1L, 2L]);
    }

    [Test]
    [Arguments("F32", 4)]
    [Arguments("F16", 2)]
    [Arguments("BF16", 2)]
    public async Task ImportRejectsNonFiniteWeightBeforeReturningModel(string dtype, int bytesPerElement)
    {
        using var fixture = new SafeTensorConversionFixture();
        var values = dtype == "F32" ? SafeTensorConversionFixture.Singles(float.NaN, 1) :
            SafeTensorConversionFixture.Halves(dtype == "F16" ? (ushort)0x7C00 : (ushort)0x7F80, 0);
        fixture.Write(new SafeTensorConversionTensor("weight", dtype, [1, 2], bytesPerElement, values));
        await Assert.That(() => Import(fixture)).Throws<SourceEncodingException>();
    }

    [Test]
    [Arguments("I32", 4)]
    [Arguments("I64", 8)]
    [Arguments("U8", 1)]
    [Arguments("BOOL", 1)]
    public async Task ImportRejectsUnsupportedWeightRecipes(string dtype, int bytesPerElement)
    {
        using var fixture = new SafeTensorConversionFixture();
        fixture.Write(new SafeTensorConversionTensor("weight", dtype, [1, 2], bytesPerElement, []));
        await Assert.That(() => Import(fixture)).Throws<NotSupportedException>();
    }

    [Test]
    [Arguments("schema")]
    [Arguments("unknown")]
    [Arguments("nested-unknown")]
    [Arguments("duplicate")]
    [Arguments("null-inputs")]
    [Arguments("null-outputs")]
    [Arguments("null-nodes")]
    [Arguments("null-node")]
    [Arguments("null-shape")]
    [Arguments("missing-weight")]
    public async Task ImportRejectsMalformedOrIncompleteGraphSidecar(string kind)
    {
        using var fixture = new SafeTensorConversionFixture();
        fixture.Write(new SafeTensorConversionTensor("weight", "F32", [1, 2], 4, []));
        var graph = SafeTensorConversionFixture.DefaultGraph;
        graph = kind switch
        {
            "schema" => graph.Replace("\"schema_version\":1", "\"schema_version\":2", StringComparison.Ordinal),
            "unknown" => graph.Replace("\"schema_version\":1", "\"schema_version\":1,\"execute\":\"remote.py\"", StringComparison.Ordinal),
            "nested-unknown" => graph.Replace("\"name\":\"linear\"", "\"name\":\"linear\",\"code\":\"remote.py\"", StringComparison.Ordinal),
            "duplicate" => graph.Replace("\"schema_version\":1", "\"schema_version\":2,\"schema_version\":1", StringComparison.Ordinal),
            "null-inputs" => /*lang=json,strict*/ "{\"schema_version\":1,\"inputs\":null,\"outputs\":[\"y\"],\"nodes\":[]}",
            "null-outputs" => /*lang=json,strict*/ "{\"schema_version\":1,\"inputs\":[],\"outputs\":null,\"nodes\":[]}",
            "null-nodes" => /*lang=json,strict*/ "{\"schema_version\":1,\"inputs\":[],\"outputs\":[],\"nodes\":null}",
            "null-node" => /*lang=json,strict*/ "{\"schema_version\":1,\"inputs\":[],\"outputs\":[\"y\"],\"nodes\":[null]}",
            "null-shape" => graph.Replace(/*lang=json,strict*/ "[{\"symbol\":null,\"minimum\":2,\"maximum\":2}]", "null", StringComparison.Ordinal),
            "missing-weight" => graph.Replace("\"weight\"", "\"missing\"", StringComparison.Ordinal),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        File.WriteAllText(fixture.Graph, graph, Encoding.UTF8);
        await Assert.That(() => Import(fixture)).Throws<InvalidDataException>();
    }

    [Test]
    [Arguments("tensor-elements")]
    [Arguments("aggregate-elements")]
    [Arguments("tensor-count")]
    [Arguments("source-bytes")]
    [Arguments("graph-bytes")]
    public async Task ImportRejectsBoundedResourceOverflowBeforeDecode(string kind)
    {
        using var fixture = new SafeTensorConversionFixture();
        SafeTensorConversionTensor[] tensors = kind switch
        {
            "tensor-elements" => [new SafeTensorConversionTensor("weight", "F32", [1_000_001], 4, [])],
            "aggregate-elements" => [.. Enumerable.Range(0, 17).Select(index => new SafeTensorConversionTensor(index == 0 ? "weight" : $"w{index}", "F32", [1_000_000], 4, []))],
            "tensor-count" => [.. Enumerable.Range(0, 1025).Select(index => new SafeTensorConversionTensor(index == 0 ? "weight" : $"w{index}", "F32", [1], 4, []))],
            _ => [new SafeTensorConversionTensor("weight", "F32", [1, 2], 4, [])],
        };
        fixture.Write(tensors);
        if (kind == "source-bytes")
        {
            using var stream = File.OpenWrite(fixture.Source);
            stream.SetLength((128L * 1024 * 1024) + 1);
        }
        if (kind == "graph-bytes")
        {
            File.WriteAllText(fixture.Graph, new string(' ', (4 * 1024 * 1024) + 1), Encoding.UTF8);
        }
        await Assert.That(() => Import(fixture)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task CancellationIsObservedBeforeOpeningSourceFiles()
    {
        using var fixture = new SafeTensorConversionFixture();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.That(() => Import(fixture, cancellation.Token)).Throws<OperationCanceledException>();
    }

    private static ConversionModel Import(SafeTensorConversionFixture fixture, CancellationToken cancellationToken = default) =>
        SafeTensorsModelImporter.Import(fixture.Source, fixture.Graph, cancellationToken);
}
