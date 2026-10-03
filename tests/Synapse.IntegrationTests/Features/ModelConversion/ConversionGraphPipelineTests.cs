using ManagedCode.Synapse.Runtime.Features.ModelConversion;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

public sealed class ConversionGraphPipelineTests
{
    [Test]
    public async Task RemovesInternalIdentityAndUnreachableWorkWithoutChangingOutputs()
    {
        var model = Model([Input("x", 2)], ["y"],
            [new("alias", "Identity", ["x"], "alias_value"),
             new("live", "Linear", ["alias_value", "w"], "y"),
             new("dead", "Silu", ["unused"], "dead_value")],
            [new("w", [1, 2], [2, 3]), new("unused", [1], [7])]);

        var prepared = ConversionGraphPipeline.Prepare(model);

        await Assert.That(prepared.Graph.Nodes.Length).IsEqualTo(1);
        await Assert.That(prepared.Graph.Nodes[0].Inputs).IsEquivalentTo(["x", "w"]);
        await Assert.That(prepared.Graph.Outputs).IsEquivalentTo(["y"]);
        await Assert.That(prepared.Tensors.Select(tensor => tensor.Name)).IsEquivalentTo(["w"]);
        await Assert.That(model.Graph.Nodes[1].Inputs[0]).IsEqualTo("alias_value");
    }

    [Test]
    public async Task ExecutesRealLinearBatchWithBias()
    {
        var model = Model([BatchInput("x", "batch", 1, 4, 2)], ["y"],
            [new("linear", "Linear", ["x", "w", "b"], "y")],
            [new("w", [2, 2], [2, 1, -1, 3]), new("b", [2], [1, -1])]);

        var output = ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [1, 2, 3, 4] },
            new Dictionary<string, long[]> { ["x"] = [2, 2] });

        await Assert.That(output["y"]).IsEquivalentTo([5f, 4f, 11f, 8f]);
    }

    [Test]
    public async Task BindsOneDynamicGraphAtTwoBatchSizesAndAppliesSoftmaxPerRow()
    {
        var model = Model([BatchInput("x", "batch", 1, 4, 2), BatchInput("z", "batch", 1, 4, 2)], ["probability"],
            [new("add", "Add", ["x", "z"], "sum"),
             new("softmax", "Softmax", ["sum"], "probability")], []);
        foreach (var batch in new[] { 1, 3 })
        {
            var output = ConversionGraphPipeline.Execute(model,
                new Dictionary<string, float[]>
                {
                    ["x"] = [.. Enumerable.Range(0, batch * 2).Select(index => (float)((index % 2) + 1))],
                    ["z"] = new float[batch * 2],
                },
                new Dictionary<string, long[]> { ["x"] = [batch, 2], ["z"] = [batch, 2] });

            await Assert.That(output["probability"].Length).IsEqualTo(batch * 2);
            for (var row = 0; row < batch; row++)
            {
                await Assert.That(Math.Abs(output["probability"][row * 2] - (1 / (1 + Math.Exp(1)))) < 1e-6).IsTrue();
                await Assert.That(Math.Abs(output["probability"].Skip(row * 2).Take(2).Sum() - 1) < 1e-6).IsTrue();
            }
        }
    }

    [Test]
    public async Task RejectsInconsistentSymbolBindingsBeforeEvaluatingNonFiniteInputs()
    {
        var model = Model([BatchInput("x", "batch", 1, 4, 2), BatchInput("z", "batch", 1, 4, 2)], ["y"],
            [new("add", "Add", ["x", "z"], "y")], []);

        await Assert.That(() => ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [float.NaN, 1], ["z"] = [1, 2, 3, 4] },
            new Dictionary<string, long[]> { ["x"] = [1, 2], ["z"] = [2, 2] }))
            .Throws<ArgumentException>();
    }

    [Test]
    public async Task RejectsBatchOutsideDeclaredBounds()
    {
        var model = Model([BatchInput("x", "batch", 1, 2, 2)], ["y"], [new("silu", "Silu", ["x"], "y")], []);

        await Assert.That(() => ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = new float[6] },
            new Dictionary<string, long[]> { ["x"] = [3, 2] })).Throws<ArgumentException>();
    }

    [Test]
    public async Task RejectsBroadcastingAndUnsupportedOperations()
    {
        var broadcast = Model([Input("x", 2), Input("z", 1)], ["y"], [new("add", "Add", ["x", "z"], "y")], []);
        var unsupported = Model([Input("x", 2)], ["y"], [new("gelu", "Gelu", ["x"], "y")], []);

        await Assert.That(() => ConversionGraphPipeline.Prepare(broadcast)).Throws<InvalidDataException>();
        await Assert.That(() => ConversionGraphPipeline.Prepare(unsupported)).Throws<NotSupportedException>();
    }

    [Test]
    public async Task RejectsCyclesAndNonFiniteWeights()
    {
        var cycle = Model([Input("x", 2)], ["y"],
            [new("first", "Silu", ["y"], "z"), new("second", "Silu", ["z"], "y")], []);
        var nonFinite = Model([Input("x", 2)], ["y"], [new("linear", "Linear", ["x", "w"], "y")],
            [new("w", [1, 2], [float.PositiveInfinity, 0])]);

        await Assert.That(() => ConversionGraphPipeline.Prepare(cycle)).Throws<InvalidDataException>();
        await Assert.That(() => ConversionGraphPipeline.Prepare(nonFinite)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task PreservesPublicIdentityNamesAndCopiesInputPayloads()
    {
        var model = Model([Input("x", 2)], ["first", "second"],
            [new("identity_one", "Identity", ["x"], "first"), new("identity_two", "Identity", ["first"], "second")], []);
        float[] input = [-0f, 4];

        var result = ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = input },
            new Dictionary<string, long[]> { ["x"] = [2] });
        input[1] = 99;
        result["first"][1] = 88;

        await Assert.That(result["second"][1]).IsEqualTo(4f);
        await Assert.That(BitConverter.SingleToInt32Bits(result["second"][0])).IsEqualTo(BitConverter.SingleToInt32Bits(-0f));
    }

    [Test]
    public async Task ExecutesBatchConstantsMultiplyAndSilu()
    {
        var model = Model([new("x", [new(null, 2, 2), new(null, 2, 2)])], ["y"],
            [new("multiply", "Multiply", ["x", "scale"], "scaled"), new("silu", "Silu", ["scaled"], "y")],
            [new("scale", [2, 2], [1, 2, 3, 4])]);
        var result = ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [1, -1, 2, -2] },
            new Dictionary<string, long[]> { ["x"] = [2, 2] });
        double[] scaled = [1, -2, 6, -8];

        for (var index = 0; index < scaled.Length; index++)
        {
            var expected = scaled[index] / (1 + Math.Exp(-scaled[index]));
            await Assert.That(Math.Abs(result["y"][index] - expected) < 1e-6).IsTrue();
        }
    }

    [Test]
    public async Task RejectsSymbolicFeatureAxesAndOversizedShapes()
    {
        var featureSymbol = Model([new("x", [new("features", 1, 8)])], ["y"], [new("silu", "Silu", ["x"], "y")], []);
        var oversized = Model([new("x", [new(null, 1001, 1001), new(null, 1000, 1000)])], ["y"],
            [new("silu", "Silu", ["x"], "y")], []);

        await Assert.That(() => ConversionGraphPipeline.Prepare(featureSymbol)).Throws<InvalidDataException>();
        await Assert.That(() => ConversionGraphPipeline.Prepare(oversized)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task OrdersAcyclicNodesAndRejectsConflictingProducers()
    {
        var unordered = Model([Input("x", 1)], ["y"],
            [new("last", "Silu", ["intermediate"], "y"), new("first", "Silu", ["x"], "intermediate")], []);
        var duplicate = unordered with
        {
            Graph = unordered.Graph with { Nodes = [new("first", "Silu", ["x"], "y"), new("last", "Silu", ["x"], "y")] },
        };

        var prepared = ConversionGraphPipeline.Prepare(unordered);

        await Assert.That(prepared.Graph.Nodes.Select(node => node.Name)).IsEquivalentTo(["first", "last"]);
        await Assert.That(() => ConversionGraphPipeline.Prepare(duplicate)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task RejectsAggregateAllocationBeyondThePackageLimit()
    {
        var inputs = Enumerable.Range(0, 17).Select(index => Input($"x{index}", 1_000_000)).ToArray();
        var model = Model(inputs, ["y"], [new("silu", "Silu", ["x0"], "y")], []);

        await Assert.That(() => ConversionGraphPipeline.Prepare(model)).Throws<InvalidDataException>();
    }

    [Test]
    public async Task HonorsCancellationBeforePreparationOrExecution()
    {
        var model = Model([Input("x", 1)], ["y"], [new("silu", "Silu", ["x"], "y")], []);
        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.That(() => ConversionGraphPipeline.Prepare(model, cancellation.Token)).Throws<OperationCanceledException>();
        await Assert.That(() => ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = [1] },
            new Dictionary<string, long[]> { ["x"] = [1] }, cancellation.Token)).Throws<OperationCanceledException>();
    }

    private static ConversionInput Input(string name, long width) => new(name, [new(null, width, width)]);

    private static ConversionInput BatchInput(string name, string symbol, long minimum, long maximum, long width) =>
        new(name, [new(symbol, minimum, maximum), new(null, width, width)]);

    private static ConversionModel Model(ConversionInput[] inputs, string[] outputs, ConversionNode[] nodes, ConversionTensor[] tensors) =>
        new("native", new(1, inputs, outputs, nodes), tensors);
}
