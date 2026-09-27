namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal sealed record GgufTensorInfo(
    string Name,
    ulong[] Dimensions,
    uint Type,
    long Offset,
    long ByteLength)
{
    public long ElementCount => Dimensions.Aggregate(1L, static (count, dimension) =>
        checked(count * checked((long)dimension)));
}
