namespace ManagedCode.Synapse.Runtime.Features.CpuKernels;

/// <summary>Instruction-set variants of the managed Q8_0 kernel.</summary>
internal enum Q8KernelIsa
{
    Portable,
    Avx2,
    AvxVnni,
    ArmDotProduct,
}

/// <summary>C# SIMD Q8_0 kernel pinned to one explicit instruction set.</summary>
internal sealed unsafe class ManagedQ8Kernel : Q8MatrixKernel
{
    public ManagedQ8Kernel(Q8KernelIsa isa)
    {
        if (!SupportedIsas.Contains(isa))
        {
            throw new NotSupportedException(
                $"Managed Q8_0 kernel ISA '{isa}' is not supported by this CPU/runtime. " +
                $"Supported: {string.Join(", ", SupportedIsas)}.");
        }

        Isa = isa;
        Name = "managed-" + isa switch
        {
            Q8KernelIsa.ArmDotProduct => "arm64-sdot",
            Q8KernelIsa.AvxVnni => "x64-avx-vnni",
            Q8KernelIsa.Avx2 => "x64-avx2",
            Q8KernelIsa.Portable => "portable-vector128",
            _ => throw new ArgumentOutOfRangeException(nameof(isa)),
        };
    }

    /// <summary>ISAs usable on this process, portable first.</summary>
    public static IReadOnlyList<Q8KernelIsa> SupportedIsas { get; } = DetectSupportedIsas();

    public Q8KernelIsa Isa { get; }

    public override string Name { get; }

    /// <summary>Creates the fastest supported variant: sdot, then AVX-VNNI, then AVX2, then portable.</summary>
    public static ManagedQ8Kernel CreateBest()
    {
        Q8KernelIsa[] preference = [Q8KernelIsa.ArmDotProduct, Q8KernelIsa.AvxVnni, Q8KernelIsa.Avx2];
        return new ManagedQ8Kernel(preference.FirstOrDefault(SupportedIsas.Contains, Q8KernelIsa.Portable));
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
        switch (Isa)
        {
            case Q8KernelIsa.ArmDotProduct:
                Q8ArmKernel.Multiply(matrix, rowStart, rowCount, activations, tokenCount, output, outputStride);
                break;
            case Q8KernelIsa.AvxVnni:
            case Q8KernelIsa.Avx2:
                Q8X86Kernel.Multiply(
                    matrix,
                    rowStart,
                    rowCount,
                    activations,
                    tokenCount,
                    output,
                    outputStride,
                    Isa == Q8KernelIsa.AvxVnni);
                break;
            case Q8KernelIsa.Portable:
            default:
                Q8PortableKernel.Multiply(matrix, rowStart, rowCount, activations, tokenCount, output, outputStride);
                break;
        }
    }

    private static Q8KernelIsa[] DetectSupportedIsas()
    {
        var isas = new List<Q8KernelIsa> { Q8KernelIsa.Portable };
        if (Q8X86Kernel.IsAvx2Supported)
        {
            isas.Add(Q8KernelIsa.Avx2);
        }

        if (Q8X86Kernel.IsVnniSupported)
        {
            isas.Add(Q8KernelIsa.AvxVnni);
        }

        if (Q8ArmKernel.IsSupported)
        {
            isas.Add(Q8KernelIsa.ArmDotProduct);
        }

        return [.. isas];
    }
}
