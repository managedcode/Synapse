using System.Runtime.InteropServices;
using System.Text;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.GpuKernels;

/// <summary>
/// The Rust <c>synapse_gpu</c> library behind its versioned C ABI (ADR-012). It is loaded only from an explicit
/// directory; a missing library, an ABI mismatch, or a missing device fails with <see cref="NotSupportedException"/>.
/// </summary>
internal sealed unsafe class NativeGpuLibrary
{
    public const uint ExpectedAbiVersion = 2;
    private const int StatusOk = 0;
    private const int StatusUnavailable = 3;
    private const int StatusOutOfMemory = 5;
    private static readonly Lock LoadGate = new();
    private static NativeGpuLibrary? _applicationLibrary;

    private readonly delegate* unmanaged<uint, NativeGpuDeviceInfo*, int> _probe;
    private readonly delegate* unmanaged<byte*, nuint, nuint> _lastError;
    private readonly delegate* unmanaged<uint, NativeDecoderDesc*, nint*, int> _create;
    private readonly delegate* unmanaged<nint, BatchToken*, nuint, float*, nuint, int> _forward;
    private readonly delegate* unmanaged<nint, void> _destroy;
    private readonly delegate* unmanaged<nint, ulong> _kvBytes;
    private readonly delegate* unmanaged<nint, uint, uint, int> _reserve;

    private NativeGpuLibrary(string path, nint handle)
    {
        LibraryPath = path;
        _probe = (delegate* unmanaged<uint, NativeGpuDeviceInfo*, int>)Export(handle, path, "synapse_gpu_probe");
        _lastError = (delegate* unmanaged<byte*, nuint, nuint>)Export(handle, path, "synapse_gpu_last_error");
        _create = (delegate* unmanaged<uint, NativeDecoderDesc*, nint*, int>)Export(handle, path, "synapse_gpu_decoder_create");
        _forward = (delegate* unmanaged<nint, BatchToken*, nuint, float*, nuint, int>)
            Export(handle, path, "synapse_gpu_decoder_forward");
        _destroy = (delegate* unmanaged<nint, void>)Export(handle, path, "synapse_gpu_decoder_destroy");
        _kvBytes = (delegate* unmanaged<nint, ulong>)Export(handle, path, "synapse_gpu_decoder_kv_bytes");
        _reserve = (delegate* unmanaged<nint, uint, uint, int>)Export(handle, path, "synapse_gpu_decoder_reserve");
    }

    public string LibraryPath { get; }

    public static string LibraryFileName =>
        OperatingSystem.IsWindows() ? "synapse_gpu.dll"
        : OperatingSystem.IsMacOS() ? "libsynapse_gpu.dylib"
        : "libsynapse_gpu.so";

    public static NativeGpuLibrary LoadFromApplicationDirectory()
    {
        lock (LoadGate)
        {
            return _applicationLibrary ??= Load(AppContext.BaseDirectory);
        }
    }

    public static NativeGpuLibrary Load(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        var path = Path.Combine(Path.GetFullPath(directory), LibraryFileName);
        if (!File.Exists(path))
        {
            throw Unavailable(path, "the library file is missing");
        }

        if (!NativeLibrary.TryLoad(path, out var handle))
        {
            throw Unavailable(path, "the operating system could not load it");
        }

        var abiVersion = ((delegate* unmanaged<uint>)Export(handle, path, "synapse_gpu_abi_version"))();
        return abiVersion == ExpectedAbiVersion
            ? new NativeGpuLibrary(path, handle)
            : throw Unavailable(path, $"its ABI version {abiVersion} differs from the required {ExpectedAbiVersion}");
    }

    public GpuDeviceInfo Probe(KernelBackend backend)
    {
        NativeGpuDeviceInfo info;
        ThrowIfFailed(_probe(BackendCode(backend), &info), backend);
        var name = new ReadOnlySpan<byte>(info.Name, 128);
        var length = name.IndexOf((byte)0);
        return new GpuDeviceInfo(
            KernelBackendNames.ToName(backend),
            Encoding.UTF8.GetString(length < 0 ? name : name[..length]),
            checked((long)info.RecommendedWorkingSetBytes),
            checked((long)info.MaxBufferBytes),
            info.UnifiedMemory != 0,
            checked((int)info.AppleFamily));
    }

    public nint CreateDecoder(KernelBackend backend, in NativeDecoderDesc description)
    {
        nint model;
        fixed (NativeDecoderDesc* pointer = &description)
        {
            ThrowIfFailed(_create(BackendCode(backend), pointer, &model), backend);
        }

        return model;
    }

    public void Forward(nint model, ReadOnlySpan<BatchToken> tokens, Span<float> logits)
    {
        fixed (BatchToken* tokenPointer = tokens)
        fixed (float* logitsPointer = logits)
        {
            ThrowIfFailed(
                _forward(model, tokenPointer, (nuint)tokens.Length, logitsPointer, (nuint)logits.Length),
                backend: null);
        }
    }

    /// <summary>Sizes a KV slot for <paramref name="positions"/> at once (ADR-017).</summary>
    public void Reserve(nint model, int slot, int positions) =>
        ThrowIfFailed(_reserve(model, checked((uint)slot), checked((uint)positions)), backend: null);

    public void Destroy(nint model) => _destroy(model);

    /// <summary>Bytes of K and V the model currently holds (ADR-017).</summary>
    public long KvBytes(nint model) => checked((long)_kvBytes(model));

    private static uint BackendCode(KernelBackend backend) => backend switch
    {
        KernelBackend.Metal => 1,
        KernelBackend.Cuda => 2,
        KernelBackend.Reference or KernelBackend.Managed or KernelBackend.Native or _ => throw new ArgumentOutOfRangeException(nameof(backend), backend, "Not a GPU backend."),
    };

    private void ThrowIfFailed(int status, KernelBackend? backend)
    {
        if (status == StatusOk)
        {
            return;
        }

        var message = LastError();
        var name = backend is { } value ? KernelBackendNames.ToName(value) : "GPU";
        throw status switch
        {
            StatusUnavailable => new NotSupportedException($"The {name} backend is unavailable: {message}"),
            StatusOutOfMemory => new InsufficientMemoryException($"The {name} backend ran out of device memory: {message}"),
            _ => new InvalidOperationException($"The {name} backend failed with status {status}: {message}"),
        };
    }

    private string LastError()
    {
        var length = checked((int)_lastError(null, 0));
        var buffer = new byte[length];
        fixed (byte* pointer = buffer)
        {
            _ = _lastError(pointer, (nuint)length);
        }

        return Encoding.UTF8.GetString(buffer);
    }

    private static nint Export(nint handle, string path, string name) =>
        NativeLibrary.TryGetExport(handle, name, out var address)
            ? address
            : throw Unavailable(path, $"it does not export '{name}'");

    private static NotSupportedException Unavailable(string path, string reason) => new(
        $"The GPU backend library is unavailable because {reason}: '{path}'. " +
        "Build it with `cargo build --release --locked -p synapse-gpu` or choose a CPU backend.");
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeGpuDeviceInfo
{
    public fixed byte Name[128];
    public ulong RecommendedWorkingSetBytes;
    public ulong MaxBufferBytes;
    public uint UnifiedMemory;
    public uint AppleFamily;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeDecoderLayerOffsets
{
    public ulong AttentionNorm;
    public ulong Query;
    public ulong Key;
    public ulong Value;
    public ulong QueryBias;
    public ulong KeyBias;
    public ulong ValueBias;
    public ulong QueryNorm;
    public ulong KeyNorm;
    public ulong AttentionOutput;
    public ulong FeedForwardNorm;
    public ulong Gate;
    public ulong Up;
    public ulong Down;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct NativeDecoderDesc
{
    public byte* Weights;
    public ulong WeightsLength;
    public NativeDecoderLayerOffsets* Layers;
    public float* RopeCosines;
    public float* RopeSines;
    public ulong TokenEmbedding;
    public ulong OutputNorm;
    public ulong Output;
    public uint LayerCount;
    public uint Hidden;
    public uint FeedForward;
    public uint Heads;
    public uint KeyValueHeads;
    public uint HeadDimension;
    public uint Vocabulary;
    public uint Context;
    public uint SessionSlots;
    public uint StepTokens;
    public uint LogitsRows;
    public float RmsEpsilon;
    public uint MatrixEncoding;
    public uint RopeLayout;
    public uint Activation;
    public uint KvPrecision;
    public uint KvGrowthPositions;
    public uint Pad;
}
