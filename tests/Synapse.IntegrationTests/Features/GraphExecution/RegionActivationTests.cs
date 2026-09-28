using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;
using static ManagedCode.Synapse.IntegrationTests.Features.GraphExecution.GraphTestModelFactory;
using static ManagedCode.Synapse.IntegrationTests.Features.GraphExecution.RegionActivationTestGraphFactory;

namespace ManagedCode.Synapse.IntegrationTests.Features.GraphExecution;

public sealed class RegionActivationTests
{
    [Test]
    public async Task AlwaysActiveMustBeNotSkippable()
    {
        var graph = WithActivation(
            CreateLinearGraph(Vector(3)),
            new RegionActivation(
                new AlwaysActive(),
                new StructuralProvenance(),
                new OutputsAbsent()));

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "always-active");
    }

    [Test]
    public async Task TrainedRouteRequiresDecisionValue()
    {
        var graph = WithActivation(
            CreateLinearGraph(Vector(3)),
            new RegionActivation(
                new ProfileDecision("trained-without-value"),
                new TrainedPolicyProvenance(Hash('a')),
                new OutputsAbsent()));

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "trained policy");
    }

    [Test]
    public async Task DecisionProducerOutsideRegion()
    {
        var graph = WithActivation(
            CreateLinearGraph(Vector(3)),
            new RegionActivation(
                new PredicateDecision(new ValueId(3), RouteScope.Step),
                new ProgrammedProvenance(),
                new OutputsAbsent()));

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "outside");
    }

    [Test]
    public async Task NonCausalDecodeRouteRejected()
    {
        var graph = CreateRouteGraph();

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "non-causal");
    }

    [Test]
    public async Task DecisionProducerMustPrecedeRegion()
    {
        var graph = CreateRouteGraph(
            TopKRouteAxis.Feature,
            routeAfterRegion: true);

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "precede");
    }

    [Test]
    public async Task AbsentOutputRequiresTolerantConsumer()
    {
        var graph = CreateAbsentOutputGraph();

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "absent output");
    }

    [Test]
    public async Task BypassShapeMustMatch()
    {
        var graph = WithActivation(
            CreateLinearGraph(Vector(3)),
            new RegionActivation(
                new PredicateDecision(new ValueId(1), RouteScope.Step),
                new ProgrammedProvenance(),
                new BypassOutputs([new ValueBypass(new ValueId(3), new ValueId(1))])));

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "bypass");
    }

    [Test]
    public async Task SkippableStateWriterRequiresHoleAwareReaders()
    {
        var graph = CreateSkippableStateWriterGraph();

        var result = ModelGraphVerifier.Verify(graph);

        await AssertInvalidRegion(result, "position holes");
    }

    private static async Task AssertInvalidRegion(GraphVerificationResult result, string message) =>
        await Assert.That(result.Diagnostics.Any(item =>
            item.Code == GraphDiagnosticCode.InvalidRegion &&
            item.Message.Contains(message, StringComparison.OrdinalIgnoreCase))).IsTrue();
}
