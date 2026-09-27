using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using static ManagedCode.Synapse.IntegrationTests.Features.GraphExecution.GraphTestModelFactory;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class TypedGraphIrTests
{
    private static readonly NumericType Fp32 = new(
        StorageDataType.Fp32,
        ComputeDataType.Fp32,
        AccumulatorDataType.Fp32);
    private static readonly NumericType I32 = new(
        StorageDataType.I32,
        ComputeDataType.I32,
        AccumulatorDataType.I32);

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
    public async Task SelfDependencyRejectedAsCycle()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var value = new GraphValue(new ValueId(1), shape, Fp32);
        var node = new GraphNode(
            new NodeId(1),
            GraphOperationKind.Add,
            [value.Id, value.Id],
            [value.Id]);
        var graph = CreateGraph([value], [node], [value.Id], [value.Id]);

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.IllegalCycle &&
            item.NodeId == node.Id)).IsTrue();
    }

    [Test]
    public async Task UnorderedStateWritersRejected()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var input = new GraphValue(new ValueId(1), shape, Fp32);
        var position = new GraphValue(
            new ValueId(2),
            new TensorShape(ShapeDimension.Fixed(1)),
            I32);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [input.Id]),
            new GraphNode(new NodeId(5), GraphOperationKind.Input, outputs: [position.Id]),
            new GraphNode(
                new NodeId(2),
                GraphOperationKind.StateAppend,
                inputs: [input.Id, position.Id],
                stateWrites: [new StateSlotId(1)]),
            new GraphNode(
                new NodeId(3),
                GraphOperationKind.StateAppend,
                inputs: [input.Id, position.Id],
                stateWrites: [new StateSlotId(1)]),
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [input.Id]),
        };
        var slot = new StateSlotDescriptor(
            new StateSlotId(1),
            new TensorShape(ShapeDimension.Fixed(8), ShapeDimension.Fixed(4)),
            Fp32,
            HasInitialValue: true);
        var graph = CreateGraph([input, position], nodes, [input.Id, position.Id], [input.Id], [slot]);

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
            requiredWeights: null,
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
            graph.Regions.Append(overlappingRegion),
            graph.Weights);

        var result = ModelGraphVerifier.Verify(graphWithOverlap);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidRegion &&
            item.NodeId == new NodeId(3))).IsTrue();
    }

    [Test]
    public async Task EntryPlumbingNotRegionMember()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var source = graph.Regions.Single();
        var invalidRegion = new RegionDescriptor(
            source.Id,
            source.Nodes.Prepend(new NodeId(1)),
            source.Inputs,
            source.Outputs,
            source.RequiredWeights,
            source.StateReads,
            source.StateWrites,
            source.Eligibility,
            source.SemanticAnnotations);
        var invalidGraph = ReplaceRegion(graph, invalidRegion);

        var result = ModelGraphVerifier.Verify(invalidGraph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidRegion &&
            item.NodeId == new NodeId(1) &&
            item.Message.Contains("plumbing", StringComparison.Ordinal))).IsTrue();
    }

    [Test]
    public async Task RegionBoundaryDerivedAndCompared()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var source = graph.Regions.Single();
        var lyingRegion = new RegionDescriptor(
            source.Id,
            source.Nodes,
            inputs: [],
            outputs: [],
            requiredWeights: [],
            stateReads: source.StateReads,
            stateWrites: source.StateWrites,
            eligibility: source.Eligibility,
            semanticAnnotations: source.SemanticAnnotations);
        var invalidGraph = ReplaceRegion(graph, lyingRegion);

        var result = ModelGraphVerifier.Verify(invalidGraph);

        var diagnostic = result.Diagnostics.Single(item =>
            item.Code == GraphDiagnosticCode.RegionBoundaryMismatch);
        await Assert.That(diagnostic.Message).Contains("inputs");
        await Assert.That(diagnostic.Message).Contains("outputs");
        await Assert.That(diagnostic.Message).Contains("required weights");
    }

    [Test]
    public async Task ConstantRequiresImmutableTensorIdentity()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var constant = graph.Nodes.Single(node => node.Operation == GraphOperationKind.Constant);
        var unboundConstant = new GraphNode(
            constant.Id,
            constant.Operation,
            constant.Inputs,
            constant.Outputs);
        var invalidGraph = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes.Select(node => node.Id == constant.Id ? unboundConstant : node),
            graph.StateSlots,
            graph.EntryPoints,
            graph.Regions,
            graph.Weights);

        var result = ModelGraphVerifier.Verify(invalidGraph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidTensorBinding &&
            item.NodeId == constant.Id)).IsTrue();
    }

    [Test]
    public async Task ConstantRequiresWeightDescriptor()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var invalidGraph = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes,
            graph.StateSlots,
            graph.EntryPoints,
            graph.Regions,
            weights: null);

        var result = ModelGraphVerifier.Verify(invalidGraph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidWeightDescriptor &&
            item.NodeId == new NodeId(2))).IsTrue();
    }

    [Test]
    public async Task OperationAttributesRequired()
    {
        var shape = new TensorShape(ShapeDimension.Fixed(4));
        var input = new GraphValue(new ValueId(1), shape, Fp32);
        var weights = new GraphValue(new ValueId(2), shape, Fp32);
        var output = new GraphValue(new ValueId(3), shape, Fp32);
        var nodes = new[]
        {
            new GraphNode(new NodeId(1), GraphOperationKind.Input, outputs: [input.Id]),
            new GraphNode(
                new NodeId(2),
                GraphOperationKind.Constant,
                outputs: [weights.Id],
                tensor: new TensorId(1)),
            new GraphNode(
                new NodeId(3),
                GraphOperationKind.RmsNorm,
                [input.Id, weights.Id],
                [output.Id]),
            new GraphNode(new NodeId(4), GraphOperationKind.Output, inputs: [output.Id]),
        };
        var graph = CreateGraph([input, weights, output], nodes, [input.Id], [output.Id]);

        var result = ModelGraphVerifier.Verify(graph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidOperationAttributes &&
            item.NodeId == new NodeId(3))).IsTrue();
    }

    [Test]
    public async Task UnrelatedOperationAttributesRejected()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var linear = graph.Nodes.Single(node => node.Operation == GraphOperationKind.Linear);
        var attributedLinear = new GraphNode(
            linear.Id,
            linear.Operation,
            linear.Inputs,
            linear.Outputs,
            linear.StateReads,
            linear.StateWrites,
            tensor: linear.Tensor,
            attributes: new NormalizationAttributes(1e-5f));
        var invalidGraph = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes.Select(node => node.Id == linear.Id ? attributedLinear : node),
            graph.StateSlots,
            graph.EntryPoints,
            graph.Regions,
            graph.Weights);

        var result = ModelGraphVerifier.Verify(invalidGraph);

        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidOperationAttributes &&
            item.NodeId == linear.Id)).IsTrue();
    }

    [Test]
    public async Task CanonicalFingerprintStableAndSensitive()
    {
        var graph = CreateLinearGraph(new TensorShape(ShapeDimension.Fixed(3)));
        var region = graph.Regions.Single();
        var reorderedRegion = new RegionDescriptor(
            region.Id,
            region.Nodes.Reverse(),
            region.Inputs.Reverse(),
            region.Outputs.Reverse(),
            region.RequiredWeights.Reverse(),
            region.StateReads.Reverse(),
            region.StateWrites.Reverse(),
            region.Eligibility,
            region.SemanticAnnotations.Reverse());
        var reordered = ReplaceRegion(graph, reorderedRegion);
        var linear = graph.Nodes.Single(node => node.Operation == GraphOperationKind.Linear);
        var changedNode = new GraphNode(
            linear.Id,
            GraphOperationKind.QuantizedLinear,
            linear.Inputs,
            linear.Outputs,
            linear.StateReads,
            linear.StateWrites,
            linear.EffectInputs,
            linear.EffectOutputs,
            linear.MergeMode,
            linear.Loop,
            linear.Tensor,
            linear.Attributes);
        var changed = new ModelGraph(
            graph.GraphVersion,
            graph.OpSetVersion,
            graph.Values,
            graph.Nodes.Select(node => node.Id == linear.Id ? changedNode : node),
            graph.StateSlots,
            graph.EntryPoints,
            graph.Regions,
            graph.Weights);

        var fingerprint = ModelGraphFingerprint.Compute(graph);

        await Assert.That(fingerprint).IsEqualTo(ModelGraphFingerprint.Compute(reordered));
        await Assert.That(fingerprint).IsNotEqualTo(ModelGraphFingerprint.Compute(changed));
        await Assert.That(fingerprint.Value.Length).IsEqualTo(64);
        await Assert.That(fingerprint.Value.All(character =>
            character is (>= '0' and <= '9') or (>= 'a' and <= 'f'))).IsTrue();
    }

}
