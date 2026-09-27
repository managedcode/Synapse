using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class TypedGraphIrTests
{
    private static readonly NumericType Fp32 = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);

    [Test]
    public async Task ValidLinearRegionPassesVerification()
    {
        var graph = CreateLinearGraph(outputShape: new TensorShape(ShapeDimension.Fixed(3)));

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.IsValid).IsTrue();
        await Assert.That(result.Diagnostics).IsEmpty();
    }

    [Test]
    public async Task ShapeMismatchRejectedWithNodeIdentity()
    {
        var graph = CreateLinearGraph(outputShape: new TensorShape(ShapeDimension.Fixed(4)));

        var result = ModelGraphVerifier.Verify(graph);

        var diagnostic = result.Diagnostics.Single(item => item.Code == GraphDiagnosticCode.ShapeMismatch);
        await Assert.That(diagnostic.NodeId).IsEqualTo(new NodeId(3));
    }

    [Test]
    public async Task IllegalCycleRejected()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var values = new[]
        {
            new GraphValue(new ValueId(1), shape, Fp32),
            new GraphValue(new ValueId(2), shape, Fp32),
        };
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Add, [new ValueId(2), new ValueId(2)], [new ValueId(1)]),
            new GraphNode(new NodeId(2), GraphOperationKind.Add, [new ValueId(1), new ValueId(1)], [new ValueId(2)]),
        };
        var graph = CreateGraph(values, nodes, [new ValueId(1)], [new ValueId(2)]);

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.Diagnostics.Any(item => item.Code == GraphDiagnosticCode.IllegalCycle)).IsTrue();
    }

    [Test]
    public async Task UnorderedStateWritersRejected()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var input = new GraphValue(new ValueId(1), shape, Fp32);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [input.Id]),
            new GraphNode(
                new NodeId(2),
                GraphOperationKind.StateAppend,
                inputs: [input.Id],
                stateWrites: [new StateSlotId(1)]),
            new GraphNode(
                new NodeId(3),
                GraphOperationKind.StateAppend,
                inputs: [input.Id],
                stateWrites: [new StateSlotId(1)]),
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [input.Id]),
        };
        var slot = new StateSlotDescriptor(new StateSlotId(1), shape, Fp32, HasInitialValue: true);
        var graph = CreateGraph([input], nodes, [input.Id], [input.Id], [slot]);

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.UnorderedStateWriters)).IsTrue();
    }

    [Test]
    public async Task RegionPredicateMustBeScalarBoolean()
    {
        var graph = CreateLinearGraph(
            outputShape: new TensorShape(ShapeDimension.Fixed(3)),
            conditionalPredicate: new ValueId(1));

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.NumericTypeMismatch &&
            item.Message.Contains("predicate", StringComparison.Ordinal))).IsTrue();
    }

    [Test]
    public async Task GraphSnapshotsCallerOwnedCollections()
    {
        var inputIds = new[] { new ValueId(1) };
        var outputIds = new[] { new ValueId(3) };
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var entryPoint = new GraphEntryPoint(new EntryPointId(7), "snapshot", inputIds, outputIds);

        inputIds[0] = new ValueId(99);
        outputIds[0] = new ValueId(100);

        await Assert.That(entryPoint.Inputs[0]).IsEqualTo(new ValueId(1));
        await Assert.That(entryPoint.Outputs[0]).IsEqualTo(new ValueId(3));
        await Assert.That(graph.Values is Array).IsFalse();
        await Assert.That(graph.Nodes is Array).IsFalse();
    }

    [Test]
    public async Task NodeCannotBelongToMultipleRegions()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var overlappingRegion = new RegionDescriptor(
            new RegionId(2),
            [new NodeId(3)],
            [new ValueId(1), new ValueId(2)],
            [new ValueId(3)],
            requiredWeights: [new TensorId(1)],
            stateReads: null,
            stateWrites: null,
            new AlwaysRequiredEligibility());
        var graphWithOverlap = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes,
            graph.StateSlots,
            graph.EntryPoints,
            graph.Regions.Append(overlappingRegion));

        var result = ModelGraphVerifier.Verify(graphWithOverlap);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidRegion &&
            item.NodeId == new NodeId(3))).IsTrue();
    }

    private static ModelGraph CreateLinearGraph(
        TensorShape outputShape,
        ValueId? conditionalPredicate = null)
    {
        var input = new GraphValue(
            new ValueId(1),
            new TensorShape(ShapeDimension.Fixed(4)),
            Fp32);
        var weights = new GraphValue(
            new ValueId(2),
            new TensorShape(ShapeDimension.Fixed(3), ShapeDimension.Fixed(4)),
            Fp32);
        var output = new GraphValue(new ValueId(3), outputShape, Fp32);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [input.Id]),
            new GraphNode(new NodeId(2), GraphOperationKind.Constant, outputs: [weights.Id]),
            new GraphNode(new NodeId(3), GraphOperationKind.Linear, [input.Id, weights.Id], [output.Id]),
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [output.Id]),
        };
        return CreateGraph(
            [input, weights, output],
            nodes,
            [input.Id],
            [output.Id],
            eligibility: conditionalPredicate is { } predicate
                ? new GraphPredicateEligibility(predicate)
                : new AlwaysRequiredEligibility());
    }

    private static ModelGraph CreateGraph(
        IEnumerable<GraphValue> values,
        IEnumerable<GraphNode> nodes,
        IReadOnlyList<ValueId> inputs,
        IReadOnlyList<ValueId> outputs,
        IEnumerable<StateSlotDescriptor>? stateSlots = null,
        ExecutionEligibility? eligibility = null)
    {
        var nodeArray = nodes.ToArray();
        var entryPoint = new GraphEntryPoint(new EntryPointId(1), "forward", inputs, outputs);
        var region = new RegionDescriptor(
            new RegionId(1),
            nodeArray.Select(node => node.Id),
            inputs,
            outputs,
            requiredWeights: [new TensorId(1)],
            stateReads: null,
            stateWrites: stateSlots?.Select(slot => slot.Id),
            eligibility ?? new AlwaysRequiredEligibility(),
            semanticAnnotations: ["test"]);
        return new ModelGraph(
            new GraphVersion(1, 0),
            new OpSetVersion(1, 0),
            values,
            nodeArray,
            stateSlots,
            [entryPoint],
            [region]);
    }
}
