using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

/// <summary>Prepares a bounded native graph and executes its portable reference path.</summary>
public static class ConversionGraphPipeline
{
    /// <summary>Validates shapes and removes internal identities and unreachable pure work.</summary>
    public static ConversionModel Prepare(ConversionModel model, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var (ordered, _) = ConversionGraphValidation.Validate(model);
        var prepared = ConversionGraphOptimizer.Normalize(model, ordered);
        cancellationToken.ThrowIfCancellationRequested();
        var (_, shapes) = ConversionGraphValidation.Validate(prepared);
        var row = ConversionRowGraph.Build(prepared, shapes);
        var verification = ModelGraphVerifier.Verify(row.Graph);
        if (!verification.IsValid)
        {
            throw new InvalidDataException($"Invalid prepared Model IR: {string.Join("; ", verification.Diagnostics.Select(item => item.Message))}");
        }

        cancellationToken.ThrowIfCancellationRequested();
        return prepared;
    }

    /// <summary>Executes explicitly bound vector or independent batch inputs through verified Model IR.</summary>
    public static IReadOnlyDictionary<string, float[]> Execute(
        ConversionModel model,
        IReadOnlyDictionary<string, float[]> inputs,
        IReadOnlyDictionary<string, long[]> shapes,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(shapes);
        var prepared = Prepare(model, cancellationToken);
        var (_, declarations) = ConversionGraphValidation.Validate(prepared);
        var (bound, rows) = Bind(prepared, inputs, shapes, declarations);
        var copied = CopyInputs(prepared, inputs, bound);
        var rowGraph = ConversionRowGraph.Build(prepared, declarations);
        var results = prepared.Graph.Outputs.ToDictionary(name => name,
            name => new float[ConversionGraphValidation.Count(bound[name])], StringComparer.Ordinal);
        for (var row = 0; row < rows; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var rowInputs = prepared.Graph.Inputs.ToDictionary(input => rowGraph.Values[input.Name],
                input => Slice(copied[input.Name], bound[input.Name], row));
            var rowWeights = rowGraph.Weights.ToDictionary(weight => weight.Id,
                weight => weight.Matrix ? (float[])weight.Tensor.Data.Clone() : Slice(weight.Tensor.Data, weight.Tensor.Shape, row));
            var output = GraphReferenceInterpreter.Execute(rowGraph.Graph, new EntryPointId(1), rowInputs, rowWeights, cancellationToken);
            Store(prepared.Graph.Outputs, bound, rowGraph.Values, results, output, row);
        }

        cancellationToken.ThrowIfCancellationRequested();
        return results;
    }

    private static (Dictionary<string, long[]> Shapes, int Rows) Bind(ConversionModel model,
        IReadOnlyDictionary<string, float[]> inputs, IReadOnlyDictionary<string, long[]> supplied,
        Dictionary<string, ConversionDimension[]> declarations)
    {
        if (inputs.Count != model.Graph.Inputs.Length || supplied.Count != model.Graph.Inputs.Length)
        {
            throw new ArgumentException("Input payload and shape IDs must exactly match the graph inputs.");
        }

        var symbols = new Dictionary<string, long>(StringComparer.Ordinal);
        var rows = 1;
        foreach (var input in model.Graph.Inputs)
        {
            if (!inputs.ContainsKey(input.Name) || !supplied.TryGetValue(input.Name, out var actual) ||
                actual is null || actual.Length != input.Shape.Length)
            {
                throw new ArgumentException($"Input '{input.Name}' requires its explicit shape and payload.");
            }

            BindInput(input, actual, symbols);
            if (actual.Length == 2)
            {
                rows = checked((int)actual[0]);
            }
        }

        var bound = declarations.ToDictionary(item => item.Key,
            item => item.Value.Select(dimension => dimension.Symbol is { } symbol ? symbols[symbol] : dimension.Maximum).ToArray(),
            StringComparer.Ordinal);
        ValidateRows(model, bound, rows);
        return (bound, rows);
    }

    private static void BindInput(ConversionInput input, long[] actual, Dictionary<string, long> symbols)
    {
        for (var index = 0; index < actual.Length; index++)
        {
            var dimension = input.Shape[index];
            if (actual[index] < dimension.Minimum || actual[index] > dimension.Maximum)
            {
                throw new ArgumentException($"Input '{input.Name}' axis {index} is outside its declared bounds.");
            }

            if (dimension.Symbol is { } symbol)
            {
                if (symbols.TryGetValue(symbol, out var prior) && prior != actual[index])
                {
                    throw new ArgumentException($"Symbol '{symbol}' has inconsistent input bindings.");
                }

                symbols[symbol] = actual[index];
            }
        }
    }

    private static void ValidateRows(ConversionModel model, Dictionary<string, long[]> bound, int rows)
    {
        var rowValues = model.Graph.Inputs.Select(input => input.Name).Concat(model.Graph.Outputs)
            .Concat(model.Graph.Nodes.SelectMany(node => node.Operation == "Linear" ? node.Inputs.Take(1) : node.Inputs));
        foreach (var name in rowValues)
        {
            var shape = bound[name];
            _ = ConversionGraphValidation.Count(shape);
            if (shape.Length == 2 && shape[0] != rows)
            {
                throw new ArgumentException("All independent batch values must have one common bound row count.");
            }
        }
    }

    private static Dictionary<string, float[]> CopyInputs(ConversionModel model,
        IReadOnlyDictionary<string, float[]> inputs, Dictionary<string, long[]> shapes)
    {
        var copy = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var input in model.Graph.Inputs)
        {
            var payload = inputs[input.Name];
            if (payload is null || payload.Length != ConversionGraphValidation.Count(shapes[input.Name]) ||
                payload.Any(value => !float.IsFinite(value)))
            {
                throw new ArgumentException($"Input '{input.Name}' payload must match its shape and contain finite FP32 values.");
            }

            copy.Add(input.Name, (float[])payload.Clone());
        }

        return copy;
    }

    private static float[] Slice(float[] values, long[] shape, int row)
    {
        var width = checked((int)shape[^1]);
        var offset = shape.Length == 1 ? 0 : checked(row * width);
        return values.AsSpan(offset, width).ToArray();
    }

    private static void Store(string[] names, Dictionary<string, long[]> shapes, Dictionary<string, ValueId> ids,
        Dictionary<string, float[]> results, IReadOnlyDictionary<ValueId, float[]> output, int row)
    {
        foreach (var name in names)
        {
            var shape = shapes[name];
            if (shape.Length == 2 || row == 0)
            {
                var offset = shape.Length == 1 ? 0 : checked(row * (int)shape[^1]);
                output[ids[name]].CopyTo(results[name], offset);
            }
        }
    }
}
