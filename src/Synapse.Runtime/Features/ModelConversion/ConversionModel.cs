using System.Text.Json.Serialization;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

/// <summary>A fixed dimension or explicitly bounded source symbol.</summary>
public sealed record ConversionDimension(string? Symbol, long Minimum, long Maximum);

/// <summary>A named FP32 graph input with row-major dimensions.</summary>
public sealed record ConversionInput(string Name, ConversionDimension[] Shape);

/// <summary>A native operation with named inputs and one named output.</summary>
public sealed record ConversionNode(string Name, string Operation, string[] Inputs, string Output);

/// <summary>A bounded decoded FP32 constant; linear matrices use [out,in].</summary>
public sealed record ConversionTensor(string Name, long[] Shape, float[] Data);

/// <summary>An inert source graph sidecar for tensor-only formats.</summary>
public sealed record ConversionGraph(int SchemaVersion, ConversionInput[] Inputs, string[] Outputs, ConversionNode[] Nodes);

/// <summary>A normalized source model ready for verified graph preparation.</summary>
public sealed record ConversionModel(string SourceFormat, ConversionGraph Graph, ConversionTensor[] Tensors);

/// <summary>Explicit finite bounds for named source dimensions.</summary>
public sealed record ConversionBound(long Minimum, long Maximum);

[JsonSerializable(typeof(ConversionGraph))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
internal sealed partial class ConversionJsonContext : JsonSerializerContext;
