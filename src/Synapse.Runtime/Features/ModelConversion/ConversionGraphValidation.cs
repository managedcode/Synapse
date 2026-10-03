namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal static class ConversionGraphValidation
{
    internal const int MaximumElements = 1_000_000;
    private const int MaximumAggregateElements = 16_000_000;
    private const int MaximumDeclarations = 4096;

    internal static (ConversionNode[] Nodes, Dictionary<string, ConversionDimension[]> Shapes) Validate(ConversionModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        var graph = model.Graph;
        if (graph is null || graph.SchemaVersion != 1 || graph.Inputs is null || graph.Outputs is null ||
            graph.Nodes is null || model.Tensors is null || graph.Inputs.Length is 0 or > MaximumDeclarations ||
            graph.Outputs.Length is 0 or > MaximumDeclarations || graph.Nodes.Length > MaximumDeclarations ||
            model.Tensors.Length > 1024 || string.IsNullOrWhiteSpace(model.SourceFormat))
        {
            throw new InvalidDataException("Native graph schema or declaration count is unsupported.");
        }

        var shapes = Declarations(model);
        var ordered = Order(graph, shapes.Keys);
        var tensors = model.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        foreach (var node in ordered)
        {
            shapes.Add(node.Output, Infer(node, shapes, tensors));
        }

        if (graph.Outputs.Any(output => !Name(output) || !shapes.ContainsKey(output)) ||
            graph.Outputs.Distinct(StringComparer.Ordinal).Count() != graph.Outputs.Length)
        {
            throw new InvalidDataException("Graph outputs must be unique declared values.");
        }

        if (shapes.Values.Sum(shape => (long)Count(shape.Select(dimension => dimension.Maximum))) > MaximumAggregateElements ||
            graph.Outputs.Sum(output => (long)Count(shapes[output].Select(dimension => dimension.Maximum))) > MaximumAggregateElements)
        {
            throw new InvalidDataException($"Graph declarations or outputs exceed the {MaximumAggregateElements} aggregate element bound.");
        }

        return (ordered, shapes);
    }

    private static Dictionary<string, ConversionDimension[]> Declarations(ConversionModel model)
    {
        var shapes = new Dictionary<string, ConversionDimension[]>(StringComparer.Ordinal);
        var symbols = new Dictionary<string, ConversionDimension>(StringComparer.Ordinal);
        foreach (var input in model.Graph.Inputs)
        {
            if (input is null || !Name(input.Name) || input.Shape is null || input.Shape.Length is < 1 or > 2)
            {
                throw new InvalidDataException("Graph inputs require unique names and rank one or two.");
            }

            for (var index = 0; index < input.Shape.Length; index++)
            {
                var dimension = input.Shape[index];
                if (dimension is null || dimension.Minimum <= 0 || dimension.Maximum < dimension.Minimum ||
                    (dimension.Symbol is null && dimension.Minimum != dimension.Maximum) ||
                    (dimension.Symbol is not null && (!Name(dimension.Symbol) || index != 0 || input.Shape.Length != 2)))
                {
                    throw new InvalidDataException("Only the first batch axis may have a finite bounded symbol.");
                }

                if (dimension.Symbol is { } symbol && symbols.TryGetValue(symbol, out var prior) && prior != dimension)
                {
                    throw new InvalidDataException($"Symbol '{symbol}' has inconsistent bounds.");
                }

                if (dimension.Symbol is { } key)
                {
                    symbols[key] = dimension;
                }
            }

            _ = Count(input.Shape.Select(dimension => dimension.Maximum));
            if (!shapes.TryAdd(input.Name, [.. input.Shape]))
            {
                throw new InvalidDataException($"Value '{input.Name}' is declared more than once.");
            }
        }

        foreach (var tensor in model.Tensors)
        {
            if (tensor is null || !Name(tensor.Name) || tensor.Shape is null || tensor.Data is null ||
                tensor.Shape.Length is < 1 or > 2 || Count(tensor.Shape) != tensor.Data.Length ||
                tensor.Data.Any(value => !float.IsFinite(value)) || !shapes.TryAdd(tensor.Name,
                    [.. tensor.Shape.Select(value => new ConversionDimension(null, value, value))]))
            {
                throw new InvalidDataException("Graph constants require unique names, bounded shapes and finite FP32 data.");
            }
        }

        return shapes;
    }

    private static ConversionNode[] Order(ConversionGraph graph, IEnumerable<string> declarations)
    {
        var producers = new Dictionary<string, int>(StringComparer.Ordinal);
        var names = new HashSet<string>(StringComparer.Ordinal);
        var values = declarations.ToHashSet(StringComparer.Ordinal);
        for (var index = 0; index < graph.Nodes.Length; index++)
        {
            var node = graph.Nodes[index];
            if (node is null || !Name(node.Name) || !Name(node.Output) || node.Inputs is null ||
                node.Inputs.Any(input => !Name(input)) || !names.Add(node.Name) ||
                values.Contains(node.Output) || !producers.TryAdd(node.Output, index))
            {
                throw new InvalidDataException("Graph node names and outputs must be unique and nonempty.");
            }

            ValidateOperation(node);
        }

        var dependencies = new int[graph.Nodes.Length];
        var children = Enumerable.Range(0, graph.Nodes.Length).Select(_ => new List<int>()).ToArray();
        for (var index = 0; index < graph.Nodes.Length; index++)
        {
            foreach (var input in graph.Nodes[index].Inputs.Distinct(StringComparer.Ordinal))
            {
                if (producers.TryGetValue(input, out var producer))
                {
                    dependencies[index]++;
                    children[producer].Add(index);
                }
                else if (!values.Contains(input))
                {
                    throw new InvalidDataException($"Node '{graph.Nodes[index].Name}' consumes undeclared '{input}'.");
                }
            }
        }

        return Sort(graph.Nodes, dependencies, children);
    }

    private static ConversionNode[] Sort(ConversionNode[] nodes, int[] dependencies, List<int>[] children)
    {
        var ready = new PriorityQueue<int, int>();
        for (var index = 0; index < dependencies.Length; index++)
        {
            if (dependencies[index] == 0)
            {
                ready.Enqueue(index, index);
            }
        }

        var ordered = new List<ConversionNode>(nodes.Length);
        while (ready.TryDequeue(out var index, out _))
        {
            ordered.Add(nodes[index]);
            foreach (var child in children[index])
            {
                if (--dependencies[child] == 0)
                {
                    ready.Enqueue(child, child);
                }
            }
        }

        if (ordered.Count != nodes.Length)
        {
            throw new InvalidDataException("Native graph contains a data dependency cycle.");
        }

        return [.. ordered];
    }

    private static void ValidateOperation(ConversionNode node)
    {
        var validCount = node.Operation switch
        {
            "Linear" => node.Inputs.Length is 2 or 3,
            "Add" or "Multiply" => node.Inputs.Length == 2,
            "Identity" or "Silu" or "Softmax" => node.Inputs.Length == 1,
            _ => throw new NotSupportedException($"Node '{node.Name}' uses unsupported operation '{node.Operation}'."),
        };
        if (!validCount)
        {
            throw new InvalidDataException($"Node '{node.Name}' has an invalid input count.");
        }
    }

    private static ConversionDimension[] Infer(ConversionNode node, Dictionary<string, ConversionDimension[]> shapes,
        Dictionary<string, ConversionTensor> tensors)
    {
        var input = shapes[node.Inputs[0]];
        if (node.Operation is "Add" or "Multiply" && !input.SequenceEqual(shapes[node.Inputs[1]]))
        {
            throw new InvalidDataException($"Node '{node.Name}' requires identical input shapes; broadcasting is unsupported.");
        }

        if (node.Operation != "Linear")
        {
            return [.. input];
        }

        if (!tensors.TryGetValue(node.Inputs[1], out var weight) || weight.Shape.Length != 2 ||
            input[^1].Symbol is not null || weight.Shape[1] != input[^1].Maximum)
        {
            throw new InvalidDataException($"Node '{node.Name}' requires constant Linear weights [out,in].");
        }

        if (node.Inputs.Length == 3 && (!tensors.TryGetValue(node.Inputs[2], out var bias) ||
            bias.Shape.Length != 1 || bias.Shape[0] != weight.Shape[0]))
        {
            throw new InvalidDataException($"Node '{node.Name}' requires a constant vector bias [out].");
        }

        var result = (ConversionDimension[])input.Clone();
        result[^1] = new(null, weight.Shape[0], weight.Shape[0]);
        _ = Count(result.Select(dimension => dimension.Maximum));
        return result;
    }

    internal static int Count(IEnumerable<long> dimensions)
    {
        var count = 1L;
        foreach (var dimension in dimensions)
        {
            if (dimension <= 0 || dimension > MaximumElements || count > MaximumElements / dimension)
            {
                throw new InvalidDataException($"A tensor exceeds the {MaximumElements} element safety bound.");
            }

            count *= dimension;
        }

        return (int)count;
    }

    private static bool Name(string? name) => !string.IsNullOrWhiteSpace(name) && name.Length <= 1024;
}
