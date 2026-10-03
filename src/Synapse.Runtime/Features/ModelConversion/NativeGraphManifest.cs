using System.Text.Json.Serialization;

namespace ManagedCode.Synapse.Runtime.Features.ModelConversion;

internal sealed record NativeGraphManifest(int SchemaVersion, string SourceFormat, ConversionGraph Graph,
    string GraphFingerprint, ConversionSource[] Sources, string[] AppliedPasses, NativeGraphTensor[] Tensors, long PayloadLength);

internal sealed record NativeGraphTensor(string Name, long[] Shape, long Offset, long ByteLength);

[JsonSerializable(typeof(NativeGraphManifest))]
[JsonSerializable(typeof(ConversionPackageInfo))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, AllowDuplicateProperties = false)]
internal sealed partial class NativeGraphJsonContext : JsonSerializerContext;
