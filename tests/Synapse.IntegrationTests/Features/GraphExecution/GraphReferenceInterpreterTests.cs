using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class GraphReferenceInterpreterTests
{
    private static readonly NumericType Fp64Accumulator = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp64);

    [Test]
    public async Task ExecutesVerifiedLinearGraphFromIr()
    {
        var graph = GraphTestModelFactory.CreateLinearGraph(
            new TensorShape(ShapeDimension.Fixed(3)),
            numericType: Fp64Accumulator);
        float[] input = [1f, 2f, -1f, 0.5f];
        float[] weight = [1f, 0f, 0f, 0f, 0f, 2f, 1f, 0f, 1f, 1f, 1f, 1f];

        var result = GraphReferenceInterpreter.Execute(
            graph,
            new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [new ValueId(1)] = input },
            new Dictionary<TensorId, float[]> { [new TensorId(1)] = weight });

        input[0] = 100f;
        weight[0] = 100f;
        await Assert.That(result[new ValueId(3)]).IsEquivalentTo([1f, 3f, 2.5f]);
    }

    [Test]
    public async Task ExecutesRmsNormAndSiluChainFromIr()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var graph = GraphTestModelFactory.CreateGraph(
            [new GraphValue(new ValueId(1), shape, Fp64Accumulator),
             new GraphValue(new ValueId(2), shape, Fp64Accumulator),
             new GraphValue(new ValueId(3), shape, Fp64Accumulator),
             new GraphValue(new ValueId(4), shape, Fp64Accumulator)],
            [new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [new ValueId(1)]),
             new GraphNode(new NodeId(2), GraphOperationKind.Constant,
                 outputs: [new ValueId(2)], tensor: new TensorId(1)),
             new GraphNode(new NodeId(3), GraphOperationKind.RmsNorm,
                 [new ValueId(1), new ValueId(2)], [new ValueId(3)],
                 attributes: new NormalizationAttributes(1e-5f)),
             new GraphNode(new NodeId(4), GraphOperationKind.Silu,
                 [new ValueId(3)], [new ValueId(4)]),
             new GraphNode(new NodeId(5), GraphOperationKind.Output, inputs: [new ValueId(4)])],
            [new ValueId(1)],
            [new ValueId(4)]);
        float[] input = [0.25f, -2f, 3.5f, 0f];
        float[] weight = [1f, 0.5f, -1.25f, 2f];

        var result = GraphReferenceInterpreter.Execute(
            graph,
            new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [new ValueId(1)] = input },
            new Dictionary<TensorId, float[]> { [new TensorId(1)] = weight });

        var sumOfSquares = input.Sum(value => (double)value * value);
        var scale = 1.0 / Math.Sqrt((sumOfSquares / input.Length) + 1e-5f);
        for (var index = 0; index < input.Length; index++)
        {
            var normalized = input[index] * scale * weight[index];
            var expected = normalized / (1 + Math.Exp(-normalized));
            await Assert.That(NumericalPolicy.Fp32.IsWithinTolerance(
                result[new ValueId(4)][index], expected)).IsTrue();
        }
    }

    [Test]
    public async Task RejectsUnsupportedOperationBeforeEvaluatingInput()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(1));
        var graph = GraphTestModelFactory.CreateGraph(
            [new GraphValue(new ValueId(1), shape, Fp64Accumulator),
             new GraphValue(new ValueId(2), shape, Fp64Accumulator)],
            [new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [new ValueId(1)]),
             new GraphNode(new NodeId(2), GraphOperationKind.Gelu,
                 [new ValueId(1)], [new ValueId(2)]),
             new GraphNode(new NodeId(3), GraphOperationKind.Output, inputs: [new ValueId(2)])],
            [new ValueId(1)],
            [new ValueId(2)]);

        NotSupportedException? failure = null;
        try
        {
            _ = GraphReferenceInterpreter.Execute(
                graph,
                new EntryPointId(1),
                new Dictionary<ValueId, float[]> { [new ValueId(1)] = [float.NaN] },
                new Dictionary<TensorId, float[]>());
        }
        catch (NotSupportedException exception)
        {
            failure = exception;
        }

        await Assert.That(failure?.Message.Contains("NodeId { Value = 2 }", StringComparison.Ordinal)).IsTrue();
    }

    [Test]
    public async Task RejectsShortWeightPayloadBeforeExecution()
    {
        var graph = GraphTestModelFactory.CreateLinearGraph(
            new TensorShape(ShapeDimension.Fixed(3)),
            numericType: Fp64Accumulator);

        await Assert.That(() => GraphReferenceInterpreter.Execute(
            graph,
            new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [new ValueId(1)] = [1f, 2f, 3f, 4f] },
            new Dictionary<TensorId, float[]> { [new TensorId(1)] = [1f] }))
            .Throws<ArgumentException>();
    }

    [Test]
    public async Task RejectsUnimplementedFp32Accumulation()
    {
        var graph = GraphTestModelFactory.CreateLinearGraph(
            new TensorShape(ShapeDimension.Fixed(3)));

        await Assert.That(() => GraphReferenceInterpreter.Execute(
            graph,
            new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [new ValueId(1)] = [1f, 2f, 3f, 4f] },
            new Dictionary<TensorId, float[]> { [new TensorId(1)] = new float[12] }))
            .Throws<NotSupportedException>();
    }

    [Test]
    public async Task RejectsValidConditionalRegionBeforeExecution()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var graph = GraphTestModelFactory.CreateGraph(
            [new GraphValue(new ValueId(1), shape, Fp64Accumulator),
             new GraphValue(new ValueId(2), shape, Fp64Accumulator)],
            [new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [new ValueId(1)]),
             new GraphNode(new NodeId(2), GraphOperationKind.Silu,
                 [new ValueId(1)], [new ValueId(2)]),
             new GraphNode(new NodeId(3), GraphOperationKind.Output, inputs: [new ValueId(2)])],
            [new ValueId(1)],
            [new ValueId(2)],
            activation: new RegionActivation(
                new ProfileDecision("tiny-test"),
                new StructuralProvenance(),
                new BypassOutputs([new ValueBypass(new ValueId(2), new ValueId(1))])));

        await Assert.That(() => GraphReferenceInterpreter.Execute(
            graph,
            new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [new ValueId(1)] = [1f, 2f, 3f, 4f] },
            new Dictionary<TensorId, float[]>()))
            .Throws<NotSupportedException>();
    }
}
