namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

internal sealed record OnnxAttribute(string Name, long? Integer, float? Float);
internal sealed record OnnxNode(string Name, string Operation, string[] Inputs, string Output, Dictionary<string, OnnxAttribute> Attributes);

internal static class OnnxNodeReader
{
    public static OnnxNode Read(BoundedProtoReader reader, int ordinal)
    {
        var inputs = new List<string>();
        var outputs = new List<string>();
        var attributes = new Dictionary<string, OnnxAttribute>(StringComparer.Ordinal);
        string? name = null;
        string? operation = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            switch (field)
            {
                case 1: AddArgument(inputs, reader.Text(wire)); break;
                case 2: AddArgument(outputs, reader.Text(wire)); break;
                case 3: reader.Once(field); name = reader.Text(wire); break;
                case 4: reader.Once(field); operation = reader.Text(wire); break;
                case 5:
                    var attribute = ReadAttribute(reader.Message(wire));
                    if (attributes.Count == 8 || !attributes.TryAdd(attribute.Name, attribute))
                    {
                        throw BoundedProtoReader.Error("Duplicate or excessive ONNX attributes.");
                    }
                    break;
                case 7:
                    reader.Once(field);
                    if (reader.Text(wire) is not ("" or "ai.onnx"))
                    {
                        throw BoundedProtoReader.Error("Custom ONNX domains are unsupported.");
                    }
                    break;
                case 6 or 9: reader.Skip(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX node field, overload or device configuration.");
            }
        }
        if (string.IsNullOrWhiteSpace(operation) || outputs.Count != 1 || string.IsNullOrWhiteSpace(outputs[0]))
        {
            throw BoundedProtoReader.Error("ONNX nodes require a supported operation and one named output.");
        }
        return new(string.IsNullOrWhiteSpace(name) ? $"onnx.node.{ordinal}" : name, operation, [.. inputs], outputs[0], attributes);
    }

    private static void AddArgument(List<string> arguments, string value)
    {
        if (arguments.Count == 4)
        {
            throw BoundedProtoReader.Error("Too many ONNX node inputs or outputs.");
        }
        arguments.Add(value);
    }

    private static OnnxAttribute ReadAttribute(BoundedProtoReader reader)
    {
        string? name = null;
        long? integer = null;
        float? floating = null;
        long? type = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            reader.Once(field);
            switch (field)
            {
                case 1: name = reader.Text(wire); break;
                case 2: floating = reader.Float(wire); break;
                case 3: integer = reader.Integer(wire); break;
                case 13: reader.Skip(wire); break;
                case 20: type = reader.Integer(wire); break;
                default: throw BoundedProtoReader.Error("Only scalar ONNX float/integer attributes are supported; subgraphs are rejected.");
            }
        }
        if (string.IsNullOrWhiteSpace(name) || (integer.HasValue == floating.HasValue)
            || (integer.HasValue && type != 2) || (floating.HasValue && (type != 1 || !float.IsFinite(floating.Value))))
        {
            throw BoundedProtoReader.Error("Malformed ONNX attribute name, type or value.");
        }
        return new(name, integer, floating);
    }
}
