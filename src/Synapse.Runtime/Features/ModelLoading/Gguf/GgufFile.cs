using System.IO.MemoryMappedFiles;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal sealed unsafe class GgufFile : IDisposable, IMappedWeights
{
    private const uint Float32Type = 0;
    private readonly MemoryMappedFile _mapping;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _pointer;
    private bool _disposed;

    private GgufFile(
        string sourceFile,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyDictionary<string, GgufTensorInfo> tensors,
        IReadOnlyDictionary<string, GgufArrayReference> arrays,
        MemoryMappedFile mapping,
        MemoryMappedViewAccessor view,
        byte* pointer,
        long length)
    {
        SourceFile = sourceFile;
        Length = length;
        Metadata = metadata;
        Tensors = tensors;
        Arrays = arrays;
        _mapping = mapping;
        _view = view;
        _pointer = pointer;
    }

    public string SourceFile { get; }

    /// <summary>Bytes in the mapped file.</summary>
    public long Length { get; }

    /// <summary>Page-aligned address of file offset zero; valid until disposal.</summary>
    public byte* BasePointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _pointer;
        }
    }

    public IReadOnlyDictionary<string, object> Metadata { get; }

    public IReadOnlyDictionary<string, GgufTensorInfo> Tensors { get; }

    /// <summary>Metadata arrays located but not materialized at open (for example the tokenizer vocabulary).</summary>
    public IReadOnlyDictionary<string, GgufArrayReference> Arrays { get; }

    public static GgufFile Open(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var descriptor = GgufHeaderReader.Read(fullPath);
        var mapping = MemoryMappedFile.CreateFromFile(
            fullPath,
            FileMode.Open,
            mapName: null,
            capacity: 0,
            MemoryMappedFileAccess.Read);
        var view = mapping.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* pointer = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
        pointer += view.PointerOffset;
        return new GgufFile(
            Path.GetFileName(fullPath),
            descriptor.Metadata,
            descriptor.Tensors,
            descriptor.Arrays,
            mapping,
            view,
            pointer,
            new FileInfo(fullPath).Length);
    }

    public string GetRequiredString(string key) => Metadata.TryGetValue(key, out var value) && value is string text
            ? text
            : throw new InvalidDataException($"GGUF metadata '{key}' is missing or not a string.");

    public int GetRequiredInt32(string key)
    {
        if (!Metadata.TryGetValue(key, out var value))
        {
            throw new InvalidDataException($"GGUF metadata '{key}' is missing.");
        }

        return Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    public float GetRequiredSingle(string key)
    {
        if (!Metadata.TryGetValue(key, out var value))
        {
            throw new InvalidDataException($"GGUF metadata '{key}' is missing.");
        }

        return Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    public GgufTensorInfo GetRequiredTensor(string name) => Tensors.TryGetValue(name, out var tensor)
            ? tensor
            : throw new InvalidDataException($"GGUF tensor '{name}' is missing.");

    public byte* GetTensorPointer(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _pointer + tensor.Offset;
    }

    /// <summary>
    /// Mapped address ranges covering every tensor payload except <paramref name="excludedTensors"/>. Adjacent
    /// tensors separated only by alignment padding merge into one range.
    /// </summary>
    public IReadOnlyList<(nint Start, long Length)> GetTensorDataRanges(IReadOnlyCollection<string> excludedTensors)
    {
        ArgumentNullException.ThrowIfNull(excludedTensors);
        ObjectDisposedException.ThrowIf(_disposed, this);
        const long MaximumPadding = 64;
        var ranges = new List<(long Offset, long End)>();
        foreach (var tensor in Tensors.Values.OrderBy(tensor => tensor.Offset))
        {
            var end = tensor.Offset + tensor.ByteLength;
            if (excludedTensors.Contains(tensor.Name))
            {
                ranges.Add((end, end));
            }
            else if (ranges.Count > 0 && tensor.Offset - ranges[^1].End <= MaximumPadding && ranges[^1].Offset != ranges[^1].End)
            {
                ranges[^1] = (ranges[^1].Offset, end);
            }
            else
            {
                ranges.Add((tensor.Offset, end));
            }
        }

        return [.. ranges
            .Where(range => range.End > range.Offset)
            .Select(range => ((nint)(_pointer + range.Offset), range.End - range.Offset))];
    }

    public float[] ReadFloat32Tensor(string name)
    {
        var tensor = GetRequiredTensor(name);
        if (tensor.Type != Float32Type)
        {
            throw new NotSupportedException($"Tensor '{name}' must use F32 storage.");
        }

        var values = new float[checked((int)tensor.ElementCount)];
        new ReadOnlySpan<float>(GetTensorPointer(tensor), values.Length).CopyTo(values);
        return values;
    }

    /// <summary>Parses a string array from the mapping; every length is bounds-checked against the file.</summary>
    public string[] ReadStringArray(string key)
    {
        var reference = RequireArray(key, 8);
        var values = new string[reference.Count];
        var offset = reference.Offset;
        for (var index = 0; index < values.Length; index++)
        {
            var length = checked((long)Read<ulong>(offset));
            offset = checked(offset + sizeof(ulong));
            if (length > 16 * 1024 * 1024 || offset + length > Length)
            {
                throw new InvalidDataException($"GGUF array '{key}' element {index} is outside the file.");
            }

            values[index] = System.Text.Encoding.UTF8.GetString(_pointer + offset, (int)length);
            offset += length;
        }

        return values;
    }

    /// <summary>Copies an INT32 array from the mapping.</summary>
    public int[] ReadInt32Array(string key)
    {
        var reference = RequireArray(key, 5);
        if (checked(reference.Offset + ((long)reference.Count * sizeof(int))) > Length)
        {
            throw new InvalidDataException($"GGUF array '{key}' is outside the file.");
        }

        return new ReadOnlySpan<int>(_pointer + reference.Offset, reference.Count).ToArray();
    }

    private GgufArrayReference RequireArray(string key, uint elementType)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Arrays.TryGetValue(key, out var reference) && reference.ElementType == elementType
            ? reference
            : throw new InvalidDataException($"GGUF metadata '{key}' is missing or not an array of type {elementType}.");
    }

    private T Read<T>(long offset)
        where T : unmanaged =>
        offset + sizeof(T) <= Length
            ? System.Runtime.CompilerServices.Unsafe.ReadUnaligned<T>(_pointer + offset)
            : throw new InvalidDataException("GGUF metadata read is outside the file.");

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _pointer = null;
        _view.Dispose();
        _mapping.Dispose();
    }

}
