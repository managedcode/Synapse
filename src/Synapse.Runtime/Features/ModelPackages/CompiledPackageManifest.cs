using System.Text.Json.Serialization;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

internal sealed record CompiledPackageManifest(
    int SchemaVersion,
    string Architecture,
    string SourceFile,
    long SourceLength,
    string SourceSha256,
    string GraphFingerprint,
    long HeaderLength,
    long PayloadLength,
    CompiledTensor[] Tensors);

internal sealed record CompiledTensor(string Name, long SourceOffset, long Offset, long ByteLength, uint Encoding, ulong[] Dimensions);

/// <summary>Verified lossless compiled package identity and provenance.</summary>
public sealed record CompiledPackageInfo(
    string Identity,
    string Architecture,
    string SourceFile,
    string SourceSha256,
    string GraphFingerprint,
    int TensorCount,
    long Length);

[JsonSerializable(typeof(CompiledPackageManifest))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
internal sealed partial class CompiledPackageJsonContext : JsonSerializerContext;
