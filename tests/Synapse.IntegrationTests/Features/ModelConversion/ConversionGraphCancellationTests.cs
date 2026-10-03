using System.Diagnostics;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelConversion;

[NotInParallel]
public sealed class ConversionGraphCancellationTests
{
    // TEST-CNV-001-4: real single-row dense math observes cancellation between nodes.
    [Test]
    public async Task ScalarInterpreterCancelsWithinOneLongRow()
    {
        var model = LongRow();
        var (_, shapes) = ConversionGraphValidation.Validate(model);
        var row = ConversionRowGraph.Build(model, shapes);
        var input = Enumerable.Repeat(1f, 1000).ToArray();
        using var cancellation = new CancellationTokenSource();
        cancellation.CancelAfter(TimeSpan.FromSeconds(1));
        var elapsed = Stopwatch.StartNew();

        await Assert.That(() => GraphReferenceInterpreter.Execute(row.Graph, new EntryPointId(1),
            new Dictionary<ValueId, float[]> { [row.Values["x"]] = input },
            new Dictionary<TensorId, float[]> { [row.Weights[0].Id] = model.Tensors[0].Data }, cancellation.Token))
            .Throws<OperationCanceledException>();

        await Assert.That(elapsed.Elapsed < TimeSpan.FromSeconds(3)).IsTrue();
        await Assert.That(input.All(value => value == 1)).IsTrue();
    }

    [Test]
    public async Task ConversionForwardsCancellationDuringOneLongRow()
    {
        var model = LongRow();
        var input = Enumerable.Repeat(1f, 1000).ToArray();
        using var cancellation = new CancellationTokenSource();
        cancellation.CancelAfter(TimeSpan.FromSeconds(1));
        var elapsed = Stopwatch.StartNew();

        await Assert.That(() => ConversionGraphPipeline.Execute(model,
            new Dictionary<string, float[]> { ["x"] = input },
            new Dictionary<string, long[]> { ["x"] = [1, 1000] }, cancellation.Token))
            .Throws<OperationCanceledException>();

        await Assert.That(elapsed.Elapsed < TimeSpan.FromSeconds(3)).IsTrue();
        await Assert.That(input.All(value => value == 1)).IsTrue();
    }

    [Test]
    public async Task ScalarInterpreterObservesCancellationBeforePreflight()
    {
        var model = LongRow();
        var (_, shapes) = ConversionGraphValidation.Validate(model);
        var row = ConversionRowGraph.Build(model, shapes);
        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.That(() => GraphReferenceInterpreter.Execute(row.Graph, new EntryPointId(1),
            new Dictionary<ValueId, float[]>(), new Dictionary<TensorId, float[]>(), cancellation.Token))
            .Throws<OperationCanceledException>();
    }

    private static ConversionModel LongRow()
    {
        var weights = new float[1_000_000];
        for (var index = 0; index < 1000; index++)
        {
            weights[(index * 1000) + index] = 1;
        }

        var nodes = Enumerable.Range(0, 4096).Select(index =>
            new ConversionNode($"layer{index}", "Linear", [index == 0 ? "x" : $"value{index - 1}", "weights"], $"value{index}")).ToArray();
        return new("native", new(1,
            [new("x", [new(null, 1, 1), new(null, 1000, 1000)])], ["value4095"], nodes),
            [new("weights", [1000, 1000], weights)]);
    }
}
