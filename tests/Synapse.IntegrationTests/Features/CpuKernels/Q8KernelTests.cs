using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.CpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.IntegrationTests.Features.CpuKernels;

[NotInParallel]
public sealed class Q8KernelTests
{
    private static readonly (string Tensor, int RowStart, int RowCount)[] Cases =
    [
        ("blk.0.attn_q.weight", 0, 64),
        ("blk.3.attn_k.weight", 5, 37),
        ("blk.7.ffn_down.weight", 1, 19),
        ("output.weight", 151900, 36),
    ];

    [Test]
    public async Task ManagedKernelMatchesFp64Oracle()
    {
        using var file = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        var isas = ManagedQ8Kernel.SupportedIsas;
        await Assert.That(isas).Contains(Q8KernelIsa.Portable);
        foreach (var isa in isas)
        {
            var kernel = new ManagedQ8Kernel(isa);
            var worst = WorstAcrossCases(file, kernel);
            await Assert.That(worst).IsLessThanOrEqualTo(1.0).Because($"ISA {isa}");
        }
    }

    [Test]
    public async Task UnsupportedManagedIsaFailsExplicitly()
    {
        var unsupported = Enum.GetValues<Q8KernelIsa>().Except(ManagedQ8Kernel.SupportedIsas).ToArray();
        foreach (var isa in unsupported)
        {
            await Assert.That(() => new ManagedQ8Kernel(isa)).Throws<NotSupportedException>();
        }

        await Assert.That(ManagedQ8Kernel.SupportedIsas).Contains(ManagedQ8Kernel.CreateBest().Isa);
    }

    [Test]
    public async Task NativeKernelMatchesManagedKernel()
    {
        using var file = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        var native = NativeQ8Kernel.LoadFromApplicationDirectory();
        var managed = ManagedQ8Kernel.CreateBest();

        await Assert.That(native.AbiVersion).IsEqualTo(NativeQ8Kernel.ExpectedAbiVersion);
        await Assert.That(native.Name).StartsWith("native-");
        await Assert.That(WorstAcrossCases(file, native)).IsLessThanOrEqualTo(1.0);
        foreach (var (tensor, rowStart, rowCount) in Cases)
        {
            var matrix = Q8Matrix.FromTensor(file, file.GetRequiredTensor(tensor));
            var activations = Q8KernelOracle.CreateActivations(matrix.Columns, 3, 77);
            var nativeOutput = Q8KernelOracle.Run(native, matrix, rowStart, rowCount, activations, 3);
            var managedOutput = Q8KernelOracle.Run(managed, matrix, rowStart, rowCount, activations, 3);
            var largest = nativeOutput.Zip(managedOutput, (left, right) => Math.Abs(left - right)).Max();
            var scale = managedOutput.Max(Math.Abs);
            await Assert.That(largest).IsLessThanOrEqualTo(Math.Max(scale, 1f) * 1e-4f).Because(tensor);
        }
    }

    [Test]
    public async Task MissingNativeLibraryFailsExplicitly()
    {
        var directory = Directory.CreateTempSubdirectory("synapse-no-native-");
        try
        {
            await Assert.That(() => NativeQ8Kernel.Load(directory.FullName))
                .Throws<NotSupportedException>()
                .WithMessageContaining(directory.FullName);
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    [Test]
    public async Task KernelRejectsOutOfRangeRows()
    {
        using var file = GgufFile.Open(ReferenceBenchmarkFixture.GetModelPath());
        var matrix = Q8Matrix.FromTensor(file, file.GetRequiredTensor("blk.0.attn_k.weight"));
        var activations = Q8KernelOracle.CreateActivations(matrix.Columns, 1, 5);
        var kernel = ManagedQ8Kernel.CreateBest();

        await Assert.That(() => Q8KernelOracle.Run(kernel, matrix, matrix.Rows - 3, 4, activations, 1))
            .Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => Q8KernelOracle.Run(kernel, matrix, 0, 4, activations, 2))
            .Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => Q8Matrix.FromTensor(file, file.GetRequiredTensor("blk.0.attn_norm.weight")))
            .Throws<NotSupportedException>();
    }

    private static double WorstAcrossCases(GgufFile file, Q8MatrixKernel kernel)
    {
        var worst = 0.0;
        foreach (var (tensor, rowStart, rowCount) in Cases)
        {
            var matrix = Q8Matrix.FromTensor(file, file.GetRequiredTensor(tensor));
            foreach (var tokens in new[] { 1, 3, 5 })
            {
                var activations = Q8KernelOracle.CreateActivations(matrix.Columns, tokens, 1000 + tokens);
                var output = Q8KernelOracle.Run(kernel, matrix, rowStart, rowCount, activations, tokens);
                worst = Math.Max(
                    worst,
                    Q8KernelOracle.WorstRelativeError(matrix, rowStart, rowCount, activations, tokens, output));
            }
        }

        return worst;
    }
}
