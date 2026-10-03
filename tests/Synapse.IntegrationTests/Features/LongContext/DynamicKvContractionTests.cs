using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary>REQ-CTX-010: request-sized KV contraction preserves exact values and respects session ownership.</summary>
[NotInParallel]
public sealed class DynamicKvContractionTests
{
    private static readonly int[] LongPrompt = [.. Enumerable.Range(0, 257).Select(index => index % 64)];
    private static readonly int[] ShortPrompt = [.. LongPrompt.Take(31)];

    [Test]
    public async Task CpuReservationContractsAtFourfoldBoundaryAndKeepsPrefixBits()
    {
        var cache = new Qwen2KvCache(layers: 2, maximumPositions: 512, kvWidth: 4, growthPositions: 64);
        float[] key = [1.25f, -2.5f, float.Epsilon, -0f];
        float[] value = [3.75f, -4.5f, -float.Epsilon, 0f];
        cache.Reserve(512);
        cache.Store(1, 127, key, value);
        cache.Reserve(129);
        await Assert.That(cache.AllocatedPositions).IsEqualTo(512);

        cache.Reserve(128);

        await Assert.That(cache.AllocatedPositions).IsEqualTo(128);
        await Assert.That(Bits(cache.GetKey(1, 127, 0, 4).ToArray()).SequenceEqual(Bits(key))).IsTrue();
        await Assert.That(Bits(cache.GetValue(1, 127, 0, 4).ToArray()).SequenceEqual(Bits(value))).IsTrue();
        cache.Reserve(65);
        await Assert.That(cache.AllocatedPositions).IsEqualTo(128);
    }

    [Test]
    [Arguments(-1)]
    [Arguments(513)]
    public async Task ReservationRejectsBoundsOutsideInstanceContext(int positions)
    {
        var cache = new Qwen2KvCache(layers: 1, maximumPositions: 512, kvWidth: 4, growthPositions: 64);
        await Assert.That(() => cache.Reserve(positions)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    [Arguments(KernelBackend.Reference, KvCachePrecision.Fp32)]
    [Arguments(KernelBackend.Managed, KvCachePrecision.Fp32)]
    [Arguments(KernelBackend.Native, KvCachePrecision.Fp32)]
    [Arguments(KernelBackend.Metal, KvCachePrecision.Fp32)]
    [Arguments(KernelBackend.Metal, KvCachePrecision.Fp16)]
    public async Task LongThenShortContractsAndMatchesColdLogitsBitwise(KernelBackend backend, KvCachePrecision precision)
    {
        RequireHardware(backend);
        using var fixture = await PreparedFixture.CreateAsync();
        using var warm = fixture.Load(backend, precision);
        using var cold = fixture.Load(backend, precision);
        _ = warm.EvaluatePromptLogits(LongPrompt);
        var longBytes = warm.AllocatedKvBytes;

        var actual = warm.EvaluateIncrementalLogits(ShortPrompt, [9, 8, 7, 6]);
        var expected = cold.EvaluateIncrementalLogits(ShortPrompt, [9, 8, 7, 6]);

        Console.WriteLine($"kv-contraction {backend}/{precision}: long={longBytes}, short={warm.AllocatedKvBytes}, cold={cold.AllocatedKvBytes}");
        await Assert.That(Bits(actual).SequenceEqual(Bits(expected))).IsTrue();
        await Assert.That(warm.AllocatedKvBytes).IsEqualTo(cold.AllocatedKvBytes);
        await Assert.That(warm.AllocatedKvBytes).IsLessThan(longBytes);
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task ReusedPrefixRemainsValidAfterContraction(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await PreparedFixture.CreateAsync();
        using var warm = fixture.Load(backend, reuse: true);
        using var cold = fixture.Load(backend);
        _ = warm.Generate(LongPrompt, 4);
        var longBytes = warm.AllocatedKvBytes;

        var actual = warm.Generate(ShortPrompt, 4);
        var expected = cold.Generate(ShortPrompt, 4);

        await Assert.That(actual.ReusedPromptTokens).IsEqualTo(ShortPrompt.Length - 1);
        await Assert.That(actual.GeneratedTokens.SequenceEqual(expected.GeneratedTokens)).IsTrue();
        await Assert.That(warm.AllocatedKvBytes).IsEqualTo(cold.AllocatedKvBytes);
        await Assert.That(warm.AllocatedKvBytes).IsLessThan(longBytes);
    }

    [Test]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task ReassignedAsyncSlotContractsAndKeepsDirectSession(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await PreparedFixture.CreateAsync();
        using var warm = fixture.Load(backend, reuse: true);
        using var cold = fixture.Load(backend);
        _ = warm.Generate(ShortPrompt, 4);
        _ = await warm.GenerateAsync(LongPrompt, 4, CancellationToken.None);
        var longBytes = warm.AllocatedKvBytes;

        var actual = await warm.GenerateAsync(ShortPrompt, 4, CancellationToken.None);
        var expected = await cold.GenerateAsync(ShortPrompt, 4, CancellationToken.None);
        var direct = warm.Generate(ShortPrompt, 4);

        Console.WriteLine($"kv-async-contraction {backend}: long={longBytes}, short={warm.AllocatedKvBytes}");
        await Assert.That(actual.GeneratedTokens.SequenceEqual(expected.GeneratedTokens)).IsTrue();
        await Assert.That(warm.AllocatedKvBytes).IsEqualTo(2 * cold.AllocatedKvBytes);
        await Assert.That(warm.AllocatedKvBytes).IsLessThan(longBytes);
        await Assert.That(direct.ReusedPromptTokens).IsEqualTo(ShortPrompt.Length - 1);
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task DisposedModelReleasesItsReferencedKvArrays(KernelBackend backend)
    {
        RequireHardware(backend);
        using var fixture = await PreparedFixture.CreateAsync();
        using var model = fixture.Load(backend);
        _ = model.Generate(LongPrompt, 4);
        var before = model.AllocatedKvBytes;

        model.Dispose();

        Console.WriteLine($"kv-disposal {backend}: before={before}, after={model.AllocatedKvBytes}");
        await Assert.That(before).IsGreaterThan(0);
        await Assert.That(model.AllocatedKvBytes).IsEqualTo(0);
    }

    private static int[] Bits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];

    private static void RequireHardware(KernelBackend backend)
    {
        if (backend == KernelBackend.Metal)
        {
            _ = GpuHardware.RequireMetal();
        }
    }

    internal sealed record PreparedFixture(string SourcePath, string PackagePath) : IDisposable
    {
        public static async Task<PreparedFixture> CreateAsync()
        {
            var source = TinyQwen2Gguf.Write(new(2, 2, 1, 64, 128, 64, 512), seed: 20261003);
            var package = Path.ChangeExtension(source, ".synapse");
            try
            {
                _ = await CompiledPackageCompiler.CompileAsync(source, package);
                return new(source, package);
            }
            catch
            {
                File.Delete(source);
                File.Delete(package);
                throw;
            }
        }

        public Qwen2Model Load(KernelBackend backend, KvCachePrecision precision = KvCachePrecision.Fp32, bool reuse = false) =>
            Qwen2Model.Load(PackagePath, new ModelLoadOptions
            {
                ContextSize = 512,
                MaximumParallelism = 2,
                MaximumConcurrentSessions = 1,
                KernelBackend = backend,
                KvCachePrecision = precision,
                KvGrowthPositions = 64,
                ReusePromptPrefix = reuse,
            });

        public void Dispose()
        {
            File.Delete(PackagePath);
            File.Delete(SourcePath);
        }
    }
}
