using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.Runtime.Features.ModelPackages;

internal static class CompiledPackageValidation
{
    internal static ModelGraph Graph(GgufFile file)
    {
        if (file.GetRequiredString("general.architecture") != "qwen2")
        {
            throw new NotSupportedException("Lossless compilation supports the executable Qwen2 family only.");
        }

        var dimensions = Qwen2ModelComposition.ReadDimensions(file, new ModelLoadOptions { ContextSize = 1 });
        return Qwen2ModelComposition.BuildVerifiedGraph(file, dimensions);
    }

    internal static void Source(GgufDescriptor descriptor)
    {
        if (descriptor.DataOffset is <= 0 or > CompiledPackageFormat.MaximumHeaderLength ||
            descriptor.Tensors.Count is <= 0 or > CompiledPackageFormat.MaximumTensorCount)
        {
            throw new InvalidDataException("Source GGUF header or tensor count exceeds compiled package bounds.");
        }

        var alignment = descriptor.Metadata.TryGetValue("general.alignment", out var value)
            ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : 32;
        var end = descriptor.DataOffset;
        foreach (var tensor in descriptor.Tensors.Values.OrderBy(tensor => tensor.Offset))
        {
            if (tensor.Offset < end || tensor.Offset % alignment != 0 || tensor.ByteLength <= 0 || tensor.Dimensions.Any(dimension => dimension == 0))
            {
                throw new InvalidDataException("Source GGUF tensors overlap, are misaligned, or have empty shapes.");
            }

            end = checked(tensor.Offset + tensor.ByteLength);
        }
    }

    internal static GgufTensorInfo[] OrderedTensors(GgufDescriptor descriptor, ModelGraph graph)
    {
        var offsets = graph.Weights.Select(weight => weight.Source.Offset).ToArray();
        var byOffset = descriptor.Tensors.Values.ToDictionary(tensor => tensor.Offset);
        return [.. offsets.Select(offset => byOffset[offset]),
            .. descriptor.Tensors.Values.Where(tensor => !offsets.Contains(tensor.Offset)).OrderBy(tensor => tensor.Name, StringComparer.Ordinal)];
    }

    internal static GgufDescriptor Rebase(GgufDescriptor source, CompiledPackageManifest manifest, long payloadOffset)
    {
        Source(source);
        if (source.DataOffset != manifest.HeaderLength || source.Tensors.Count != manifest.Tensors.Length ||
            (source.Metadata.GetValueOrDefault("general.architecture") as string) != manifest.Architecture)
        {
            throw new InvalidDataException("Compiled manifest and preserved source index disagree.");
        }

        var tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        var next = CompiledPackageFormat.Align(manifest.HeaderLength);
        foreach (var tensor in manifest.Tensors)
        {
            if (tensor is null || string.IsNullOrEmpty(tensor.Name) || tensor.Dimensions is null ||
                !source.Tensors.TryGetValue(tensor.Name, out var original) || tensor.SourceOffset != original.Offset ||
                tensor.ByteLength != original.ByteLength || tensor.Encoding != original.Type ||
                !tensor.Dimensions.SequenceEqual(original.Dimensions) || tensor.Offset != next ||
                checked(tensor.Offset + tensor.ByteLength) > manifest.PayloadLength || !tensors.TryAdd(tensor.Name,
                    original with { Offset = checked(payloadOffset + tensor.Offset), SourceOffset = original.Offset }))
            {
                throw new InvalidDataException("Compiled tensor index contains duplicate, overlapping, misaligned, or incompatible ranges.");
            }

            next = CompiledPackageFormat.Align(checked(tensor.Offset + tensor.ByteLength));
        }

        if (next != manifest.PayloadLength)
        {
            throw new InvalidDataException("Compiled tensor payload has an invalid final boundary.");
        }

        var arrays = source.Arrays.ToDictionary(pair => pair.Key,
            pair => pair.Value with { Offset = checked(payloadOffset + pair.Value.Offset) }, StringComparer.Ordinal);
        return new GgufDescriptor(source.Metadata, tensors, arrays, checked(payloadOffset + source.DataOffset));
    }
}
