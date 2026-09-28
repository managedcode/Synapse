using System.Runtime.InteropServices;

namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>
/// The Rust <c>synapse_kernels</c> Q8_0 kernel behind its versioned C ABI (ADR-006). The library is loaded only
/// from an explicit directory; a missing library or ABI mismatch fails with <see cref="NotSupportedException"/>.
/// </summary>
internal sealed unsafe class NativeQ8Kernel : Q8MatrixKernel
{
    public const uint ExpectedAbiVersion = 1;
    private const uint NeonDotProductCapability = 1u << 1;
    private const uint Avx2Capability = 1u << 2;

    private readonly delegate* unmanaged<byte*, nuint, nuint, nuint, sbyte*, float*, nuint, float*, nuint, int> _multiply;

    private NativeQ8Kernel(
        string libraryPath,
        uint abiVersion,
        uint capabilities,
        delegate* unmanaged<byte*, nuint, nuint, nuint, sbyte*, float*, nuint, float*, nuint, int> multiply)
    {
        LibraryPath = libraryPath;
        AbiVersion = abiVersion;
        Capabilities = capabilities;
        _multiply = multiply;
        Name = "native-" + ((capabilities & NeonDotProductCapability) != 0 ? "neon-dotprod"
            : (capabilities & Avx2Capability) != 0 ? "x64-avx2"
            : "scalar");
    }

    public string LibraryPath { get; }

    public uint AbiVersion { get; }

    public uint Capabilities { get; }

    public override string Name { get; }

    public static string LibraryFileName =>
        OperatingSystem.IsWindows() ? "synapse_kernels.dll"
        : OperatingSystem.IsMacOS() ? "libsynapse_kernels.dylib"
        : "libsynapse_kernels.so";

    public static NativeQ8Kernel LoadFromApplicationDirectory() => Load(AppContext.BaseDirectory);

    public static NativeQ8Kernel Load(string directory)
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

        var abiVersion = ((delegate* unmanaged<uint>)Export(handle, path, "synapse_kernels_abi_version"))();
        if (abiVersion != ExpectedAbiVersion)
        {
            throw Unavailable(path, $"its ABI version {abiVersion} differs from the required {ExpectedAbiVersion}");
        }

        var capabilities = ((delegate* unmanaged<uint>)Export(handle, path, "synapse_kernels_capabilities"))();
        var multiply = (delegate* unmanaged<byte*, nuint, nuint, nuint, sbyte*, float*, nuint, float*, nuint, int>)
            Export(handle, path, "synapse_q8_0_matmul");
        return new NativeQ8Kernel(path, abiVersion, capabilities, multiply);
    }

    protected override void MultiplyCore(
        in Q8Matrix matrix,
        int rowStart,
        int rowCount,
        Q8ActivationBuffer activations,
        int tokenCount,
        float* output,
        int outputStride)
    {
        var status = _multiply(
            matrix.Data + ((long)rowStart * matrix.RowBytes),
            (nuint)matrix.RowBytes,
            (nuint)rowCount,
            (nuint)matrix.BlocksPerRow,
            activations.Quants,
            activations.Scales,
            (nuint)tokenCount,
            output,
            (nuint)outputStride);
        if (status != 0)
        {
            throw new InvalidOperationException($"Native Q8_0 kernel '{LibraryPath}' returned status {status}.");
        }
    }

    private static nint Export(nint handle, string path, string name) =>
        NativeLibrary.TryGetExport(handle, name, out var address)
            ? address
            : throw Unavailable(path, $"it does not export '{name}'");

    private static NotSupportedException Unavailable(string path, string reason) => new(
        $"The native CPU kernel backend is unavailable because {reason}: '{path}'. " +
        "Build it with `cargo build --release --locked -p synapse-kernels` or choose `--backend managed`.");
}
