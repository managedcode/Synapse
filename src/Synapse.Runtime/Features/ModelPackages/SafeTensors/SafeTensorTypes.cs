namespace ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;

/// <summary>Storage encodings accepted by the bounded Safetensors reader.</summary>
public enum SafeTensorDataType
{
    /// <summary>IEEE 754 32-bit floating point.</summary>
    F32,
    /// <summary>IEEE 754 16-bit floating point.</summary>
    F16,
    /// <summary>Brain floating point 16-bit.</summary>
    BF16,
    /// <summary>Signed 64-bit integer.</summary>
    I64,
    /// <summary>Signed 32-bit integer.</summary>
    I32,
    /// <summary>Unsigned 8-bit integer.</summary>
    U8,
    /// <summary>Boolean byte.</summary>
    Boolean,
}

/// <summary>One validated tensor range in a Safetensors file.</summary>
public sealed class SafeTensorInfo
{
    internal SafeTensorInfo(
        string name,
        SafeTensorDataType dataType,
        IEnumerable<long> shape,
        long offset,
        long byteLength)
    {
        Name = name;
        DataType = dataType;
        Shape = Array.AsReadOnly([.. shape]);
        Offset = offset;
        ByteLength = byteLength;
    }

    /// <summary>Stable source tensor name.</summary>
    public string Name { get; }
    /// <summary>Physical scalar encoding.</summary>
    public SafeTensorDataType DataType { get; }
    /// <summary>Logical row-major dimensions.</summary>
    public IReadOnlyList<long> Shape { get; }
    /// <summary>Absolute byte offset in the source file.</summary>
    public long Offset { get; }
    /// <summary>Physical byte count.</summary>
    public long ByteLength { get; }
    /// <summary>Logical scalar count.</summary>
    public long ElementCount => Shape.Aggregate(1L, static (count, dimension) =>
        checked(count * dimension));
}

/// <summary>Immutable validated Safetensors index.</summary>
public sealed class SafeTensorIndex
{
    internal SafeTensorIndex(
        long fileLength,
        long dataOffset,
        IEnumerable<SafeTensorInfo> tensors,
        IReadOnlyDictionary<string, string> metadata)
    {
        FileLength = fileLength;
        DataOffset = dataOffset;
        Tensors = Array.AsReadOnly([.. tensors]);
        Metadata = new System.Collections.ObjectModel.ReadOnlyDictionary<string, string>(
            new Dictionary<string, string>(metadata, StringComparer.Ordinal));
    }

    /// <summary>Total source file length.</summary>
    public long FileLength { get; }
    /// <summary>First byte after the padded JSON header.</summary>
    public long DataOffset { get; }
    /// <summary>Validated tensors in source declaration order.</summary>
    public IReadOnlyList<SafeTensorInfo> Tensors { get; }
    /// <summary>String metadata from the optional metadata object.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }
}
