namespace ManagedCode.Synapse.Runtime.Features.ModelConversion.Onnx;

internal static class OnnxValueReader
{
    public static ConversionInput Read(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        string? name = null;
        ConversionDimension[]? shape = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            switch (field)
            {
                case 1: reader.Once(field); name = reader.Text(wire); break;
                case 2: reader.Once(field); shape = ReadType(reader.Message(wire), bounds); break;
                case 3 or 4: reader.Skip(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX value declaration field.");
            }
        }
        if (string.IsNullOrWhiteSpace(name) || shape is null)
        {
            throw BoundedProtoReader.Error("An ONNX value requires a name and FP32 tensor type.");
        }
        return new(name, shape);
    }

    private static ConversionDimension[] ReadType(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        ConversionDimension[]? shape = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            if (field == 1)
            {
                reader.Once(field);
                shape = ReadTensorType(reader.Message(wire), bounds);
            }
            else if (field == 6)
            {
                reader.Skip(wire);
            }
            else
            {
                throw BoundedProtoReader.Error("Only ONNX tensor values are supported.");
            }
        }
        return shape ?? throw BoundedProtoReader.Error("Missing ONNX tensor type.");
    }

    private static ConversionDimension[] ReadTensorType(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        long? elementType = null;
        ConversionDimension[]? shape = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            reader.Once(field);
            switch (field)
            {
                case 1: elementType = reader.Integer(wire); break;
                case 2: shape = ReadShape(reader.Message(wire), bounds); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX tensor type field.");
            }
        }
        if (elementType != 1 || shape is null)
        {
            throw BoundedProtoReader.Error("ONNX conversion requires an explicit FP32 tensor shape.");
        }
        return shape;
    }

    private static ConversionDimension[] ReadShape(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        var dimensions = new List<ConversionDimension>();
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            if (field != 1 || dimensions.Count == 2)
            {
                throw BoundedProtoReader.Error("ONNX conversion supports vector or rank-two batch shapes only.");
            }
            dimensions.Add(ReadDimension(reader.Message(wire), bounds));
        }
        if (dimensions.Count == 0 || dimensions[^1].Symbol is not null)
        {
            throw BoundedProtoReader.Error("Only the first dimension of a rank-two ONNX batch may be symbolic.");
        }
        long elements = 1;
        foreach (var dimension in dimensions)
        {
            if (dimension.Maximum > 1_000_000 / elements)
            {
                throw BoundedProtoReader.Error("An ONNX value exceeds one million elements.");
            }
            elements *= dimension.Maximum;
        }
        return [.. dimensions];
    }

    private static ConversionDimension ReadDimension(BoundedProtoReader reader, IReadOnlyDictionary<string, ConversionBound> bounds)
    {
        long? value = null;
        string? symbol = null;
        while (reader.HasData)
        {
            var (field, wire) = reader.Next();
            reader.Once(field);
            switch (field)
            {
                case 1: value = reader.Integer(wire); break;
                case 2: symbol = reader.Text(wire); break;
                case 3: reader.Skip(wire); break;
                default: throw BoundedProtoReader.Error("Unsupported ONNX dimension field.");
            }
        }
        if (value.HasValue && symbol is null && value.Value > 0)
        {
            return new(null, value.Value, value.Value);
        }
        if (!value.HasValue && !string.IsNullOrWhiteSpace(symbol) && bounds.TryGetValue(symbol, out var bound)
            && bound is not null && bound.Minimum > 0 && bound.Maximum >= bound.Minimum)
        {
            return new(symbol, bound.Minimum, bound.Maximum);
        }
        throw BoundedProtoReader.Error($"ONNX dimension '{symbol}' requires a positive static value or explicit finite bounds.");
    }
}
