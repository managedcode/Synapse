using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;

/// <summary>Executes the supported fixed-shape dense subset of verified Model IR on scalar C# operators.</summary>
public static class GraphReferenceInterpreter
{
    /// <summary>Runs one stateless FP32 entry point with explicit tensor payloads.</summary>
    /// <remarks>Inputs and weights are copied before execution; unsupported semantics fail closed in preflight.</remarks>
    public static IReadOnlyDictionary<ValueId, float[]> Execute(
        ModelGraph graph,
        EntryPointId entryPointId,
        IReadOnlyDictionary<ValueId, float[]> inputs,
        IReadOnlyDictionary<TensorId, float[]> weights)
    {
        var (entryPoint, values) = GraphReferencePreflight.Prepare(graph, entryPointId, inputs, weights);
        var declarations = graph.Values.ToDictionary(value => value.Id);
        foreach (var node in graph.Nodes)
        {
            if (node.Operation is GraphOperationKind.Input or GraphOperationKind.Constant or GraphOperationKind.Output)
            {
                continue;
            }

            var outputId = node.Outputs[0];
            var output = new float[GraphReferencePreflight.ElementCount(declarations[outputId].Shape)];
            ExecuteNode(node, values, output);
            values.Add(outputId, output);
        }

        return entryPoint.Outputs.ToDictionary(id => id, id => (float[])values[id].Clone());
    }

    private static void ExecuteNode(GraphNode node, Dictionary<ValueId, float[]> values, float[] output)
    {
        var input = values[node.Inputs[0]];
        switch (node.Operation)
        {
            case GraphOperationKind.Linear:
                ReferenceLinearOperators.Multiply(
                    input,
                    values[node.Inputs[1]],
                    node.Inputs.Count == 3 ? values[node.Inputs[2]] : [],
                    output);
                break;
            case GraphOperationKind.RmsNorm:
                ReferenceNormalizationOperators.RmsNorm(
                    input,
                    values[node.Inputs[1]],
                    ((NormalizationAttributes)node.Attributes!).Epsilon,
                    output);
                break;
            case GraphOperationKind.Add:
                ReferenceVectorOperators.Add(input, values[node.Inputs[1]], output);
                break;
            case GraphOperationKind.Multiply:
                ReferenceVectorOperators.Multiply(input, values[node.Inputs[1]], output);
                break;
            case GraphOperationKind.Silu:
                ReferenceVectorOperators.Silu(input, output);
                break;
            case GraphOperationKind.Softmax:
                ReferenceVectorOperators.Softmax(input, output);
                break;
            case GraphOperationKind.Input:
            case GraphOperationKind.Constant:
            case GraphOperationKind.Embedding:
            case GraphOperationKind.LayerNorm:
            case GraphOperationKind.QuantizedLinear:
            case GraphOperationKind.Gelu:
            case GraphOperationKind.Reshape:
            case GraphOperationKind.Transpose:
            case GraphOperationKind.Slice:
            case GraphOperationKind.Concat:
            case GraphOperationKind.Gather:
            case GraphOperationKind.Scatter:
            case GraphOperationKind.Rope:
            case GraphOperationKind.CausalAttention:
            case GraphOperationKind.StateRead:
            case GraphOperationKind.StateAppend:
            case GraphOperationKind.StateCommit:
            case GraphOperationKind.StateRollback:
            case GraphOperationKind.TopKRoute:
            case GraphOperationKind.Branch:
            case GraphOperationKind.Merge:
            case GraphOperationKind.Loop:
            case GraphOperationKind.SelectLogits:
            case GraphOperationKind.Sample:
            case GraphOperationKind.Output:
            default:
                throw new NotSupportedException($"Node {node.Id} ({node.Operation}) escaped interpreter preflight.");
        }
    }
}
