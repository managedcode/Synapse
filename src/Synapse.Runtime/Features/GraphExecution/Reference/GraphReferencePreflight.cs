using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

internal static class GraphReferencePreflight
{
    private const int MaximumTensorElements = 1_000_000;

    public static (GraphEntryPoint EntryPoint, Dictionary<ValueId, float[]> Values) Prepare(
        ModelGraph graph,
        EntryPointId entryPointId,
        IReadOnlyDictionary<ValueId, float[]> inputs,
        IReadOnlyDictionary<TensorId, float[]> weights)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(weights);

        var verification = ModelGraphVerifier.Verify(graph);
        if (!verification.IsValid)
        {
            throw new InvalidDataException(
                $"Invalid Model IR: {string.Join("; ", verification.Diagnostics.Select(diagnostic => diagnostic.Message))}");
        }

        var entryPoint = ValidateScope(graph, entryPointId);
        var values = graph.Values.ToDictionary(value => value.Id);
        var storage = new Dictionary<ValueId, float[]>();
        LoadInputs(entryPoint, values, inputs, storage);
        LoadWeights(graph, values, weights, storage);
        return (entryPoint, storage);
    }

    private static GraphEntryPoint ValidateScope(ModelGraph graph, EntryPointId entryPointId)
    {
        if (graph.EntryPoints.Count != 1 || graph.EntryPoints[0].Id != entryPointId)
        {
            throw new NotSupportedException("The scalar reference interpreter requires one selected entry point.");
        }

        if (graph.Regions.Count != 1 || graph.Regions[0].Activation.Decision is not AlwaysActive ||
            graph.StateSlots.Count != 0)
        {
            throw new NotSupportedException(
                "The scalar reference interpreter requires one always-active stateless region.");
        }

        foreach (var value in graph.Values)
        {
            var numeric = value.NumericType;
            if (numeric.Storage != StorageDataType.Fp32 || numeric.Compute != ComputeDataType.Fp32 ||
                numeric.Accumulator != AccumulatorDataType.Fp64)
            {
                throw new NotSupportedException(
                    $"Value {value.Id} requires FP32 storage/compute and FP64 reference accumulation.");
            }

            _ = ElementCount(value.Shape);
        }

        foreach (var node in graph.Nodes)
        {
            if (!Supported(node.Operation) || node.StateReads.Count != 0 || node.StateWrites.Count != 0 ||
                node.EffectInputs.Count != 0 || node.EffectOutputs.Count != 0)
            {
                throw new NotSupportedException($"Node {node.Id} ({node.Operation}) is not supported by the scalar interpreter.");
            }

            if (node.Operation == GraphOperationKind.Input &&
                !graph.EntryPoints[0].Inputs.Contains(node.Outputs[0]))
            {
                throw new NotSupportedException($"Node {node.Id} supplies an input outside the selected entry point.");
            }

            if (node.Operation is GraphOperationKind.RmsNorm or GraphOperationKind.Softmax &&
                graph.Values.Single(value => value.Id == node.Outputs[0]).Shape.Rank != 1)
            {
                throw new NotSupportedException($"Node {node.Id} requires a rank-one reference vector.");
            }
        }

        if (graph.Weights.Any(weight => weight.Encoding != WeightEncoding.Fp32))
        {
            throw new NotSupportedException("The scalar reference interpreter requires FP32 weight payloads.");
        }

        return graph.EntryPoints[0];
    }

    private static bool Supported(GraphOperationKind operation) => operation is
        GraphOperationKind.Input or GraphOperationKind.Constant or GraphOperationKind.Output or
        GraphOperationKind.Linear or GraphOperationKind.RmsNorm or GraphOperationKind.Add or
        GraphOperationKind.Multiply or GraphOperationKind.Silu or GraphOperationKind.Softmax;

    private static void LoadInputs(
        GraphEntryPoint entryPoint,
        Dictionary<ValueId, GraphValue> declarations,
        IReadOnlyDictionary<ValueId, float[]> inputs,
        Dictionary<ValueId, float[]> storage)
    {
        if (inputs.Count != entryPoint.Inputs.Count)
        {
            throw new ArgumentException("Input payload IDs must exactly match the selected entry point.", nameof(inputs));
        }

        foreach (var id in entryPoint.Inputs)
        {
            if (!inputs.TryGetValue(id, out var payload) || payload is null)
            {
                throw new ArgumentException($"Input {id} has no FP32 payload.", nameof(inputs));
            }

            storage.Add(id, CopyExact(payload, declarations[id].Shape, $"Input {id}"));
        }
    }

    private static void LoadWeights(
        ModelGraph graph,
        Dictionary<ValueId, GraphValue> declarations,
        IReadOnlyDictionary<TensorId, float[]> weights,
        Dictionary<ValueId, float[]> storage)
    {
        if (weights.Count != graph.Weights.Count)
        {
            throw new ArgumentException("Weight payload IDs must exactly match graph descriptors.", nameof(weights));
        }

        foreach (var node in graph.Nodes.Where(node => node.Operation == GraphOperationKind.Constant))
        {
            var id = node.Tensor!.Value;
            if (!weights.TryGetValue(id, out var payload) || payload is null)
            {
                throw new ArgumentException($"Tensor {id} has no FP32 payload.", nameof(weights));
            }

            var output = node.Outputs[0];
            storage.Add(output, CopyExact(payload, declarations[output].Shape, $"Tensor {id}"));
        }
    }

    private static float[] CopyExact(float[] payload, TensorShape shape, string name)
    {
        if (payload.Length != ElementCount(shape))
        {
            throw new ArgumentException($"{name} payload length does not match its fixed shape.");
        }

        return (float[])payload.Clone();
    }

    internal static int ElementCount(TensorShape shape)
    {
        var count = 1L;
        foreach (var dimension in shape.Dimensions)
        {
            if (dimension.IsSymbolic || dimension.Minimum != dimension.Maximum)
            {
                throw new NotSupportedException("The scalar reference interpreter requires fixed tensor shapes.");
            }

            count = checked(count * dimension.Maximum);
            if (count > MaximumTensorElements)
            {
                throw new NotSupportedException(
                    $"A reference tensor exceeds the {MaximumTensorElements} element safety limit.");
            }
        }

        return checked((int)count);
    }
}
