namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

internal static class OnnxTensorReader
{
    public static ConversionTensor Read(BoundedProtoReader reader)
    {
        string? name = null;
        long? dataType = null;
        var shape = new List<long>();
        var values = new List<float>();
        var raw = false;
        var typed = false;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            switch (field)
            {
                case 1: ReadDimensions(reader, wire, shape); break;
                case 2: reader.Once(field); dataType = reader.Integer(wire); break;
                case 4:
                    typed = true;
                    ReadFloats(reader, wire, values);
                    break;
                case 8: reader.Once(field); name = reader.Text(wire); break;
                case 9:
                    reader.Once(field);
                    raw = true;
                    ReadFloats(reader, wire, values, true);
                    break;
                case 12 or 16: reader.Skip(wire); break;
                case 14:
                    reader.Once(field);
                    if (reader.Integer(wire) != 0)
                    {
                        throw BoundedProtoReader.Error("External ONNX tensors are unsupported.");
                    }
                    break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX tensor field, external data or non-FP32 payload.");
            }
        }
        Validate(name, dataType, shape, values, raw, typed);
        return new(name!, [.. shape], [.. values]);
    }

    private static void ReadDimensions(BoundedProtoReader reader, int wire, List<long> shape)
    {
        var packed = wire == 2 ? reader.Message(wire) : null;
        do
        {
            if (shape.Count == 2)
            {
                throw BoundedProtoReader.Error("ONNX constants must have rank one or two.");
            }
            shape.Add((packed ?? reader).Integer(packed is null ? wire : 0));
        } while (packed?.HasData == true);
    }

    private static void ReadFloats(BoundedProtoReader reader, int wire, List<float> values, bool raw = false)
    {
        var packed = wire == 2 ? reader.Message(wire) : null;
        if (raw && packed is null)
        {
            throw BoundedProtoReader.Error("ONNX raw tensor data must be length-delimited.");
        }
        if (packed is not null && (packed.Remaining % 4 != 0 || packed.Remaining > 4_000_000))
        {
            throw BoundedProtoReader.Error("ONNX FP32 payload is misaligned or exceeds one million elements.");
        }
        if (packed is not null && !packed.HasData)
        {
            return;
        }
        do
        {
            if (values.Count == 1_000_000)
            {
                throw BoundedProtoReader.Error("ONNX tensor exceeds one million elements.");
            }
            var value = (packed ?? reader).Float(packed is null ? wire : 5);
            if (!float.IsFinite(value))
            {
                throw BoundedProtoReader.Error("ONNX FP32 constants must be finite.");
            }
            values.Add(value);
        } while (packed?.HasData == true);
    }

    private static void Validate(string? name, long? dataType, List<long> shape, List<float> values, bool raw, bool typed)
    {
        if (string.IsNullOrWhiteSpace(name) || dataType != 1 || shape.Count == 0 || raw == typed)
        {
            throw BoundedProtoReader.Error("ONNX constant needs a name, FP32 type, positive rank and exactly one data encoding.");
        }
        long elements = 1;
        foreach (var dimension in shape)
        {
            if (dimension <= 0 || dimension > 1_000_000 / elements)
            {
                throw BoundedProtoReader.Error("ONNX constant shape is invalid or exceeds one million elements.");
            }
            elements *= dimension;
        }
        if (elements != values.Count)
        {
            throw BoundedProtoReader.Error("ONNX constant data length does not match its shape.");
        }
    }
}
