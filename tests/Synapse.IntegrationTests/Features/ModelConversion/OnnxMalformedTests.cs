using ManagedCode.Synapse.Runtime.Features.ModelConversion;
using ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

public sealed class OnnxMalformedTests
{
    [Test]
    public async Task MatMulRejectsEmptyThirdInputWhileGemmAllowsOmittedBias()
    {
        using var fixture = new OnnxFixture();
        var tensor = OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4]);
        var invalid = OnnxFixture.Graph([OnnxFixture.Node("MatMul", ["x", "w", ""])], [tensor],
            [OnnxFixture.Value("x", 1, 2)], [OnnxFixture.Value("y", 1, 2)]);

        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(invalid))))
            .Throws<InvalidDataException>();

        var valid = OnnxFixture.Graph([OnnxFixture.Node("Gemm", ["x", "w", ""])], [tensor],
            [OnnxFixture.Value("x", 1, 2)], [OnnxFixture.Value("y", 1, 2)]);
        var model = OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(valid)));
        var result = ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [1, 2] },
            new Dictionary<string, long[]> { ["x"] = [1, 2] });

        await Assert.That(result["y"]).IsEquivalentTo(new float[] { 7, 10 });
    }

    [Test]
    public async Task InvalidSymbolBoundsAreReportedAsDataErrors()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [],
            [OnnxFixture.Value("x", "batch", 2)], [OnnxFixture.Value("y", "batch", 2)]);
        var path = fixture.Write(OnnxFixture.Model(graph));
        ConversionBound[] invalidBounds = [null!, new(0, 8), new(4, 3), new(1, long.MaxValue)];
        foreach (var bound in invalidBounds)
        {
            var bounds = new Dictionary<string, ConversionBound> { ["batch"] = bound };
            await Assert.That(() => OnnxModelImporter.Import(path, bounds)).Throws<InvalidDataException>();
        }
    }

    [Test]
    public async Task ContradictoryTensorPayloadsTypesAndDuplicateFieldsAreRejected()
    {
        using var fixture = new OnnxFixture();
        byte[][] invalidTensors =
        [
            OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4], true, OnnxFixture.Message(4, new byte[16])),
            OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4], true, OnnxFixture.Integer(2, 1)),
            OnnxFixture.Tensor("w", [1_000_001, 1], [1]),
            OnnxFixture.Tensor("w", [2, 2], [1, 2, 3]),
            OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4], true, OnnxFixture.Message(13, [])),
            OnnxFixture.Join(OnnxFixture.Integer(1, 2), OnnxFixture.Integer(1, 2), OnnxFixture.Integer(2, 10), OnnxFixture.Text(8, "w")),
        ];
        foreach (var tensor in invalidTensors)
        {
            var graph = OnnxFixture.Graph([OnnxFixture.Node("MatMul", ["x", "w"])], [tensor],
                [OnnxFixture.Value("x", 1, 2)], [OnnxFixture.Value("y", 1, 2)]);
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)))).Throws<InvalidDataException>();
        }
    }

    [Test]
    public async Task DuplicateSymbolsDimensionsAndOutputArityAreRejected()
    {
        using var fixture = new OnnxFixture();
        var ambiguousShape = OnnxFixture.Join(OnnxFixture.Integer(1, 2), OnnxFixture.Text(2, "batch"));
        var malformedValue = OnnxFixture.Join(OnnxFixture.Text(1, "x"), OnnxFixture.Message(2,
            OnnxFixture.Message(1, OnnxFixture.Join(OnnxFixture.Integer(1, 1), OnnxFixture.Message(2,
                OnnxFixture.Join(OnnxFixture.Message(1, ambiguousShape), OnnxFixture.Message(1, OnnxFixture.Integer(1, 2))))))));
        byte[][] badInputs = [malformedValue, OnnxFixture.Value("x", 0, 2), OnnxFixture.Value("x", 2, 2, 2), OnnxFixture.Value("x", "width")];
        foreach (var input in badInputs)
        {
            var graph = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [], [input], [OnnxFixture.Value("y", 2, 2)]);
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)))).Throws<InvalidDataException>();
        }
        var multipleOutputs = OnnxFixture.Join(OnnxFixture.Node("Identity", ["x"]), OnnxFixture.Text(2, "z"));
        var graphArity = OnnxFixture.Graph([multipleOutputs], [], [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y", 2)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graphArity)))).Throws<InvalidDataException>();
    }

    [Test]
    public async Task AttributeSubgraphsDuplicateAttributesAndNonTopologicalNodesAreRejected()
    {
        using var fixture = new OnnxFixture();
        var subgraph = OnnxFixture.Join(OnnxFixture.Text(1, "branch"), OnnxFixture.Message(6, []), OnnxFixture.Integer(20, 5));
        byte[][] invalidNodes =
        [
            OnnxFixture.Node("Identity", ["x"], "y", subgraph),
            OnnxFixture.Node("Softmax", ["x"], "y", OnnxFixture.IntAttribute("axis", -1), OnnxFixture.IntAttribute("axis", -1)),
            OnnxFixture.Node("Softmax", ["x"], "y", OnnxFixture.FloatAttribute("axis", 1)),
            OnnxFixture.Node("Identity", ["future"]),
            OnnxFixture.Node("Identity", ["x"], "x"),
        ];
        foreach (var node in invalidNodes)
        {
            var graph = OnnxFixture.Graph([node], [], [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y", 2)]);
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)))).Throws<InvalidDataException>();
        }
    }

    [Test]
    public async Task SparseInitializersOversizedGraphsAndMalformedVarintsAreRejected()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [],
            [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y", 2)], OnnxFixture.Message(15, []));
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)))).Throws<InvalidDataException>();
        var excessNodes = Enumerable.Range(0, 4097).Select(i => OnnxFixture.Node("Identity", [i == 0 ? "x" : $"y{i - 1}"], $"y{i}")).ToArray();
        var largeGraph = OnnxFixture.Graph(excessNodes, [], [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y4096", 2)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(largeGraph)))).Throws<InvalidDataException>();
        byte[][] malformed = [[0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02], [0], [0x0B], [0x3A, 0x02, 0x08]];
        foreach (var bytes in malformed)
        {
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(bytes))).Throws<InvalidDataException>();
        }
    }
}
