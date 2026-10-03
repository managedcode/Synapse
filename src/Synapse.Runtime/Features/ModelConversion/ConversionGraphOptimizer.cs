namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal static class ConversionGraphOptimizer
{
    internal static ConversionModel Normalize(ConversionModel model, ConversionNode[] ordered)
    {
        var aliases = new Dictionary<string, string>(StringComparer.Ordinal);
        var publicOutputs = model.Graph.Outputs.ToHashSet(StringComparer.Ordinal);
        var rewritten = new List<ConversionNode>(ordered.Length);
        foreach (var node in ordered)
        {
            var inputs = node.Inputs.Select(input => aliases.GetValueOrDefault(input, input)).ToArray();
            if (node.Operation == "Identity" && !publicOutputs.Contains(node.Output))
            {
                aliases.Add(node.Output, inputs[0]);
            }
            else
            {
                rewritten.Add(node with { Inputs = inputs });
            }
        }

        var reachable = Reachable(model.Graph.Outputs, rewritten);
        var nodes = rewritten.Where(node => reachable.Contains(node.Output)).ToArray();
        var usedTensors = nodes.SelectMany(node => node.Inputs).Concat(model.Graph.Outputs).ToHashSet(StringComparer.Ordinal);
        return model with
        {
            Graph = model.Graph with
            {
                Inputs = [.. model.Graph.Inputs.Select(input => input with { Shape = [.. input.Shape] })],
                Outputs = [.. model.Graph.Outputs],
                Nodes = nodes,
            },
            Tensors = [.. model.Tensors.Where(tensor => usedTensors.Contains(tensor.Name))
                .Select(tensor => tensor with { Shape = [.. tensor.Shape], Data = [.. tensor.Data] })],
        };
    }

    private static HashSet<string> Reachable(string[] outputs, List<ConversionNode> nodes)
    {
        var producers = nodes.ToDictionary(node => node.Output, StringComparer.Ordinal);
        var reachable = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Queue<string>(outputs);
        while (pending.TryDequeue(out var value))
        {
            if (reachable.Add(value) && producers.TryGetValue(value, out var producer))
            {
                foreach (var input in producer.Inputs)
                {
                    pending.Enqueue(input);
                }
            }
        }

        return reachable;
    }
}
