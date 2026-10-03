namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

internal sealed record OnnxGraph(OnnxNode[] Nodes, ConversionTensor[] Tensors, ConversionInput[] Inputs,
    ConversionInput[] Outputs, ConversionInput[] ValueInfo);

internal static class OnnxGraphReader
{
    public static OnnxGraph ReadModel(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        OnnxGraph? graph = null;
        long? irVersion = null;
        var hasOpset = false;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            switch (field)
            {
                case 1: reader.Once(field); irVersion = reader.Integer(wire); break;
                case 7: reader.Once(field); graph = ReadGraph(reader.Message(wire), bounds); break;
                case 8:
                    if (hasOpset)
                    {
                        throw BoundedProtoReader.Error("Only one default ONNX opset import is supported.");
                    }
                    ReadOpset(reader.Message(wire));
                    hasOpset = true;
                    break;
                case 2 or 3 or 4 or 5 or 6 or 14: reader.Skip(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX model field, training graph or local function.");
            }
        }
        if (graph is null || !hasOpset || irVersion is not (>= 7 and <= 10))
        {
            throw BoundedProtoReader.Error("ONNX model requires one graph, a default opset and IR version 7..10.");
        }
        return graph;
    }

    private static OnnxGraph ReadGraph(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        var nodes = new List<OnnxNode>();
        var tensors = new List<ConversionTensor>();
        var inputs = new List<ConversionInput>();
        var outputs = new List<ConversionInput>();
        var valueInfo = new List<ConversionInput>();
        long elements = 0;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            switch (field)
            {
                case 1: Add(nodes, OnnxNodeReader.Read(reader.Message(wire), nodes.Count)); break;
                case 5:
                    var tensor = OnnxTensorReader.Read(reader.Message(wire));
                    elements += tensor.Data.Length;
                    if (elements > 8_000_000)
                    {
                        throw BoundedProtoReader.Error("ONNX graph constants exceed eight million FP32 elements.");
                    }
                    Add(tensors, tensor);
                    break;
                case 11: Add(inputs, OnnxValueReader.Read(reader.Message(wire), bounds)); break;
                case 12: Add(outputs, OnnxValueReader.Read(reader.Message(wire), bounds)); break;
                case 13: Add(valueInfo, OnnxValueReader.Read(reader.Message(wire), bounds)); break;
                case 2 or 10 or 16: reader.Skip(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX graph field, quantization annotation or sparse initializer.");
            }
        }
        if (inputs.Count == 0 || outputs.Count == 0)
        {
            throw BoundedProtoReader.Error("ONNX graph requires explicit inputs and outputs.");
        }
        return new([.. nodes], [.. tensors], [.. inputs], [.. outputs], [.. valueInfo]);
    }

    private static void ReadOpset(BoundedProtoReader reader)
    {
        var domain = "";
        long? version = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            reader.Once(field);
            switch (field)
            {
                case 1: domain = reader.Text(wire); break;
                case 2: version = reader.Integer(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX opset field.");
            }
        }
        if (domain is not ("" or "ai.onnx") || version is not (>= 13 and <= 21))
        {
            throw BoundedProtoReader.Error("ONNX conversion supports only the default ai.onnx opset 13..21.");
        }
    }

    private static void Add<T>(List<T> list, T item)
    {
        if (list.Count == 4096)
        {
            throw BoundedProtoReader.Error("ONNX graph exceeds 4096 nodes, tensors or declarations.");
        }
        list.Add(item);
    }
}
