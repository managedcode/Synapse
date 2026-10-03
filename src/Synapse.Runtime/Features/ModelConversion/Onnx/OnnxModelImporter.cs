namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

/// <summary>Imports a bounded, inert ONNX graph into the native conversion model.</summary>
public static class OnnxModelImporter
{
    /// <summary>Parses supported FP32 ONNX operations without executing external code.</summary>
    public static ConversionModel Import(string path, IReadOnlyDictionary<string, ConversionBound>? bounds = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        cancellationToken.ThrowIfCancellationRequested();
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length is <= 0 or > 64L * 1024 * 1024)
        {
            throw BoundedProtoReader.Error("The initial ONNX importer requires a nonempty file no larger than 64 MiB.");
        }
        bounds ??= new Dictionary<string, ConversionBound>(StringComparer.Ordinal);
        var graph = OnnxGraphReader.ReadModel(new(stream, stream.Length, cancellationToken), bounds);
        var symbols = graph.Inputs.Concat(graph.Outputs).Concat(graph.ValueInfo)
            .SelectMany(value => value.Shape).Select(dimension => dimension.Symbol).OfType<string>().ToHashSet(StringComparer.Ordinal);
        if (bounds.Keys.Any(symbol => !symbols.Contains(symbol)))
        {
            throw BoundedProtoReader.Error("ONNX dimension bounds include an undeclared symbol.");
        }

        return Normalize(graph, cancellationToken);
    }

    private static ConversionModel Normalize(OnnxGraph graph, CancellationToken cancellationToken)
    {
        var shapes = new Dictionary<string, ConversionDimension[]>(StringComparer.Ordinal);
        var constants = new Dictionary<string, ConversionTensor>(StringComparer.Ordinal);
        foreach (var input in graph.Inputs)
        {
            AddShape(shapes, input.Name, input.Shape);
        }
        foreach (var tensor in graph.Tensors)
        {
            AddShape(shapes, tensor.Name, [.. tensor.Shape.Select(value => new ConversionDimension(null, value, value))]);
            constants.Add(tensor.Name, tensor);
        }
        var names = new HashSet<string>(graph.Inputs.Select(input => input.Name).Concat(graph.Tensors.Select(tensor => tensor.Name))
            .Concat(graph.Nodes.Select(node => node.Output)), StringComparer.Ordinal);
        var nodeNames = new HashSet<string>(StringComparer.Ordinal);
        var normalizedWeights = new Dictionary<string, string>(StringComparer.Ordinal);
        var tensors = graph.Tensors.ToList();
        var nodes = new List<ConversionNode>();
        foreach (var node in graph.Nodes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!nodeNames.Add(node.Name))
            {
                throw BoundedProtoReader.Error("Duplicate ONNX node name.");
            }
            var converted = NormalizeNode(node, shapes, constants, tensors, names, normalizedWeights);
            nodes.Add(converted);
            AddShape(shapes, node.Output, InferShape(converted, shapes));
        }
        ValidateDeclarations(graph.Outputs, shapes);
        ValidateDeclarations(graph.ValueInfo, shapes);
        return new("onnx", new(1, graph.Inputs, [.. graph.Outputs.Select(value => value.Name)], [.. nodes]), [.. tensors]);
    }

    private static ConversionNode NormalizeNode(OnnxNode node, Dictionary<string, ConversionDimension[]> shapes,
        Dictionary<string, ConversionTensor> constants, List<ConversionTensor> tensors, HashSet<string> names,
        Dictionary<string, string> normalizedWeights)
    {
        if (node.Operation is "MatMul" or "Gemm")
        {
            return NormalizeLinear(node, shapes, constants, tensors, names, normalizedWeights);
        }
        var operation = node.Operation == "Mul" ? "Multiply" : node.Operation;
        var arity = operation switch
        {
            "Add" or "Multiply" => 2,
            "Identity" or "Softmax" => 1,
            _ => throw BoundedProtoReader.Error($"Unsupported ONNX operator '{node.Operation}'."),
        };
        ValidateInputs(node, shapes, arity);
        if (operation == "Softmax")
        {
            AllowedAttributes(node, "axis");
            var axis = IntegerAttribute(node, "axis", -1);
            if (axis != -1 && axis != shapes[node.Inputs[0]].Length - 1)
            {
                throw BoundedProtoReader.Error("ONNX Softmax requires the last axis.");
            }
        }
        else
        {
            AllowedAttributes(node);
        }
        return new(node.Name, operation, node.Inputs, node.Output);
    }

    private static ConversionNode NormalizeLinear(OnnxNode node, Dictionary<string, ConversionDimension[]> shapes,
        Dictionary<string, ConversionTensor> constants, List<ConversionTensor> tensors, HashSet<string> names,
        Dictionary<string, string> normalizedWeights)
    {
        var gemm = node.Operation == "Gemm";
        var inputs = gemm && node.Inputs.Length == 3 && node.Inputs[2] == "" ? node.Inputs[..2] : node.Inputs;
        ValidateInputs(node with { Inputs = inputs }, shapes, gemm && inputs.Length == 3 ? 3 : 2);
        AllowedAttributes(node, gemm ? ["alpha", "beta", "transA", "transB"] : []);
        if (!constants.TryGetValue(inputs[1], out var source) || source.Shape.Length != 2)
        {
            throw BoundedProtoReader.Error("ONNX MatMul/Gemm requires constant rank-two B weights.");
        }
        var transposed = gemm ? IntegerAttribute(node, "transB", 0) : 0;
        if (transposed is not (0 or 1) || (gemm && (IntegerAttribute(node, "transA", 0) != 0
            || FloatAttribute(node, "alpha", 1) != 1 || FloatAttribute(node, "beta", 1) != 1
            || shapes[inputs[0]].Length != 2)))
        {
            throw BoundedProtoReader.Error("ONNX Gemm requires rank-two A, alpha=beta=1, transA=0 and transB=0 or 1.");
        }
        var inputWidth = transposed == 1 ? source.Shape[1] : source.Shape[0];
        var outputWidth = transposed == 1 ? source.Shape[0] : source.Shape[1];
        if (shapes[inputs[0]][^1].Maximum != inputWidth || (inputs.Length == 3
            && (!constants.TryGetValue(inputs[2], out var bias) || bias.Shape.Length != 1 || bias.Shape[0] != outputWidth)))
        {
            throw BoundedProtoReader.Error("ONNX Linear input or constant rank-one bias shape is incompatible.");
        }
        var name = transposed == 1 ? source.Name : GetTransposedWeight(source, tensors, names, shapes, normalizedWeights);
        var convertedInputs = (string[])inputs.Clone();
        convertedInputs[1] = name;
        return new(node.Name, "Linear", convertedInputs, node.Output);
    }

    private static string GetTransposedWeight(ConversionTensor source, List<ConversionTensor> tensors, HashSet<string> names,
        Dictionary<string, ConversionDimension[]> shapes, Dictionary<string, string> normalizedWeights)
    {
        if (normalizedWeights.TryGetValue(source.Name, out var prior))
        {
            return prior;
        }
        if (tensors.Count >= 4096 || tensors.Sum(tensor => (long)tensor.Data.Length) + source.Data.Length > 16_000_000)
        {
            throw BoundedProtoReader.Error("Normalized ONNX constants exceed 4096 tensors or sixteen million FP32 elements.");
        }
        var name = $"__synapse.onnx.linear.{tensors.Count}";
        while (!names.Add(name))
        {
            name += "_";
        }
        tensors.Add(new(name, [source.Shape[1], source.Shape[0]], Transpose(source)));
        AddShape(shapes, name, [new(null, source.Shape[1], source.Shape[1]), new(null, source.Shape[0], source.Shape[0])]);
        normalizedWeights.Add(source.Name, name);
        return name;
    }

    private static float[] Transpose(ConversionTensor source)
    {
        var rows = (int)source.Shape[0];
        var columns = (int)source.Shape[1];
        var result = new float[source.Data.Length];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                result[(column * rows) + row] = source.Data[(row * columns) + column];
            }
        }
        return result;
    }

    private static ConversionDimension[] InferShape(ConversionNode node, Dictionary<string, ConversionDimension[]> shapes)
    {
        var shape = (ConversionDimension[])shapes[node.Inputs[0]].Clone();
        if (node.Operation == "Linear")
        {
            shape[^1] = shapes[node.Inputs[1]][0];
        }
        else if (node.Operation is "Add" or "Multiply" && !shape.SequenceEqual(shapes[node.Inputs[1]]))
        {
            throw BoundedProtoReader.Error("ONNX Add/Multiply conversion requires exactly equal shapes; broadcasting is unsupported.");
        }
        return shape;
    }

    private static void ValidateInputs(OnnxNode node, Dictionary<string, ConversionDimension[]> shapes, int arity)
    {
        if (node.Inputs.Length != arity || node.Inputs.Any(input => string.IsNullOrWhiteSpace(input) || !shapes.ContainsKey(input)))
        {
            throw BoundedProtoReader.Error($"ONNX '{node.Operation}' has invalid input arity, unknown input or non-topological order.");
        }
    }

    private static void ValidateDeclarations(ConversionInput[] declarations, Dictionary<string, ConversionDimension[]> shapes)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var declaration in declarations)
        {
            if (!names.Add(declaration.Name) || !shapes.TryGetValue(declaration.Name, out var actual) || !actual.SequenceEqual(declaration.Shape))
            {
                throw BoundedProtoReader.Error("ONNX output/value_info declaration conflicts with the inferred native graph shape.");
            }
        }
    }

    private static void AddShape(Dictionary<string, ConversionDimension[]> shapes, string name, ConversionDimension[] shape)
    {
        if (!shapes.TryAdd(name, shape))
        {
            throw BoundedProtoReader.Error("ONNX input, initializer and node outputs must have unique ownership; initializer overrides are unsupported.");
        }
    }

    private static void AllowedAttributes(OnnxNode node, params string[] allowed)
    {
        if (node.Attributes.Keys.Any(name => !allowed.Contains(name, StringComparer.Ordinal)))
        {
            throw BoundedProtoReader.Error($"Unsupported ONNX attribute on '{node.Operation}'.");
        }
    }

    private static long IntegerAttribute(OnnxNode node, string name, long fallback) => node.Attributes.TryGetValue(name, out var attribute)
        ? attribute.Integer ?? throw BoundedProtoReader.Error($"ONNX attribute '{name}' requires integer type.") : fallback;

    private static float FloatAttribute(OnnxNode node, string name, float fallback) => node.Attributes.TryGetValue(name, out var attribute)
        ? attribute.Float ?? throw BoundedProtoReader.Error($"ONNX attribute '{name}' requires float type.") : fallback;
}
