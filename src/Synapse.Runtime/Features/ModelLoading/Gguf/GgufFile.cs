using System.IO.MemoryMappedFiles;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal sealed unsafe class GgufFile : IDisposable
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
        MemoryMappedFile mapping,
        MemoryMappedViewAccessor view,
        byte* pointer)
    {
        SourceFile = sourceFile;
        Metadata = metadata;
        Tensors = tensors;
        _mapping = mapping;
        _view = view;
        _pointer = pointer;
    }

    public string SourceFile { get; }

    public IReadOnlyDictionary<string, object> Metadata { get; }

    public IReadOnlyDictionary<string, GgufTensorInfo> Tensors { get; }

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
            mapping,
            view,
            pointer);
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
