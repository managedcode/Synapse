using ManagedCode.Synapse.Runtime.Features.ModelConversion;
using ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

public sealed class OnnxImportTests
{
    [Test]
    public async Task SharedConstantWeightsAreNormalizedOnce()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph(
            [OnnxFixture.Node("MatMul", ["x", "w"], "a"), OnnxFixture.Node("MatMul", ["x", "w"], "b"),
                OnnxFixture.Node("Add", ["a", "b"])],
            [OnnxFixture.Tensor("w", [2, 2], [1, 2, 3, 4])],
            [OnnxFixture.Value("x", 2)], [OnnxFixture.Value("y", 2)]);
        var model = OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)));

        await Assert.That(model.Graph.Nodes[0].Inputs[1]).IsEqualTo(model.Graph.Nodes[1].Inputs[1]);
        await Assert.That(model.Tensors.Length).IsEqualTo(2);
    }

    [Test]
    public async Task MatMulNormalizesConstantWeightsAndBoundedBatch()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph([OnnxFixture.Node("MatMul", ["x", "w"])],
            [OnnxFixture.Tensor("w", [2, 3], [1, 2, 3, 4, 5, 6])],
            [OnnxFixture.Value("x", "batch", 2)], [OnnxFixture.Value("y", "batch", 3)]);
        var model = OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)),
            new Dictionary<string, ConversionBound> { ["batch"] = new(1, 8) });

        await Assert.That(model.SourceFormat).IsEqualTo("onnx");
        await Assert.That(model.Graph.Nodes[0].Operation).IsEqualTo("Linear");
        var weights = model.Tensors.Single(x => x.Name == model.Graph.Nodes[0].Inputs[1]);
        await Assert.That(weights.Shape).IsEquivalentTo(new long[] { 3, 2 });
        await Assert.That(weights.Data).IsEquivalentTo(new float[] { 1, 4, 2, 5, 3, 6 });
        await Assert.That(model.Graph.Inputs[0].Shape[0]).IsEqualTo(new ConversionDimension("batch", 1, 8));
        var result = ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [1, 2, 3, 4] },
            new Dictionary<string, long[]> { ["x"] = [2, 2] });
        await Assert.That(result["y"]).IsEquivalentTo(new float[] { 9, 12, 15, 19, 26, 33 });
    }

    [Test]
    public async Task GemmTransposedWeightsBiasAndPackedFloatDataArePreserved()
    {
        using var fixture = new OnnxFixture();
        var node = OnnxFixture.Node("Gemm", ["x", "w", "b"], "y", OnnxFixture.IntAttribute("transB", 1));
        var graph = OnnxFixture.Graph([node],
            [OnnxFixture.Tensor("w", [3, 2], [1, 4, 2, 5, 3, 6], false), OnnxFixture.Tensor("b", [3], [7, 8, 9])],
            [OnnxFixture.Value("x", 1, 2)], [OnnxFixture.Value("y", 1, 3)]);
        var model = OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)));

        await Assert.That(model.Graph.Nodes[0].Operation).IsEqualTo("Linear");
        await Assert.That(model.Graph.Nodes[0].Inputs[2]).IsEqualTo("b");
        await Assert.That(model.Tensors.Single(x => x.Name == model.Graph.Nodes[0].Inputs[1]).Data)
            .IsEquivalentTo(new float[] { 1, 4, 2, 5, 3, 6 });
    }

    [Test]
    public async Task ElementwiseAndLastAxisSoftmaxMapWithoutChangingSemantics()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph(
            [OnnxFixture.Node("Add", ["x", "z"], "sum"), OnnxFixture.Node("Mul", ["sum", "z"], "product"),
                OnnxFixture.Node("Identity", ["product"], "copy"), OnnxFixture.Node("Softmax", ["copy"], "y", OnnxFixture.IntAttribute("axis", 1))],
            [], [OnnxFixture.Value("x", 2, 3), OnnxFixture.Value("z", 2, 3)], [OnnxFixture.Value("y", 2, 3)]);
        var model = OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph, 21)));

        await Assert.That(model.Graph.Nodes.Select(x => x.Operation).ToArray())
            .IsEquivalentTo(["Add", "Multiply", "Identity", "Softmax"]);
    }

    [Test]
    public async Task UnboundedBatchAndInconsistentOutputShapesAreRejected()
    {
        using var fixture = new OnnxFixture();
        var dynamicGraph = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [],
            [OnnxFixture.Value("x", "batch", 3)], [OnnxFixture.Value("y", "batch", 3)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(dynamicGraph))))
            .Throws<InvalidDataException>();
        var wrongOutput = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [],
            [OnnxFixture.Value("x", 2, 3)], [OnnxFixture.Value("y", 2, 4)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(wrongOutput))))
            .Throws<InvalidDataException>();
    }

    [Test]
    public async Task UnsupportedOperatorsExternalTensorsAndRecipesFailExplicitly()
    {
        using var fixture = new OnnxFixture();
        foreach (var node in new[]
        {
            OnnxFixture.Node("Relu", ["x"]), OnnxFixture.Node("Softmax", ["x"], "y", OnnxFixture.IntAttribute("axis", 0)),
            OnnxFixture.Node("Gemm", ["x", "w"], "y", OnnxFixture.FloatAttribute("alpha", 0.5f)),
            OnnxFixture.Join(OnnxFixture.Node("Identity", ["x"]), OnnxFixture.Text(7, "custom")),
        })
        {
            var graph = OnnxFixture.Graph([node], [OnnxFixture.Tensor("w", [3, 3], new float[9])],
                [OnnxFixture.Value("x", 2, 3)], [OnnxFixture.Value("y", 2, 3)]);
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(graph)))).Throws<InvalidDataException>();
        }
        var external = OnnxFixture.Graph([OnnxFixture.Node("MatMul", ["x", "w"])],
            [OnnxFixture.Tensor("w", [3, 3], new float[9], true, OnnxFixture.Integer(14, 1))],
            [OnnxFixture.Value("x", 2, 3)], [OnnxFixture.Value("y", 2, 3)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(external)))).Throws<InvalidDataException>();
    }

    [Test]
    public async Task MalformedDuplicateAndNonFiniteDataAreRejected()
    {
        using var fixture = new OnnxFixture();
        var graph = OnnxFixture.Graph([OnnxFixture.Node("Identity", ["x"])], [],
            [OnnxFixture.Value("x", 3)], [OnnxFixture.Value("y", 3)]);
        foreach (var bytes in new[]
        {
            [0x3A, 0xFF, 0xFF, 0x7F], OnnxFixture.Join(OnnxFixture.Model(graph), OnnxFixture.Message(7, graph)),
            OnnxFixture.Model(graph, 12), OnnxFixture.Model(graph, 13, OnnxFixture.Message(25, [])),
        })
        {
            await Assert.That(() => OnnxModelImporter.Import(fixture.Write(bytes))).Throws<InvalidDataException>();
        }
        var nan = OnnxFixture.Graph([OnnxFixture.Node("MatMul", ["x", "w"])],
            [OnnxFixture.Tensor("w", [3, 1], [float.NaN, 0, 0])],
            [OnnxFixture.Value("x", 3)], [OnnxFixture.Value("y", 1)]);
        await Assert.That(() => OnnxModelImporter.Import(fixture.Write(OnnxFixture.Model(nan)))).Throws<InvalidDataException>();
    }

    [Test]
    public async Task CancellationAndFileBoundsAreEnforcedBeforeDecode()
    {
        using var fixture = new OnnxFixture();
        var path = fixture.Write([]);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.That(() => OnnxModelImporter.Import(path, cancellationToken: cancelled.Token)).Throws<OperationCanceledException>();
        using (var file = new FileStream(path, FileMode.Open, FileAccess.Write))
        {
            file.SetLength((64L * 1024 * 1024) + 1);
        }
        await Assert.That(() => OnnxModelImporter.Import(path)).Throws<InvalidDataException>();
    }
}
