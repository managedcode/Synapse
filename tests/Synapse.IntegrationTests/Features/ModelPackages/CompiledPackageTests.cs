using System.Security.Cryptography;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

[NotInParallel]
public sealed class CompiledPackageTests
{
    [Test]
    public async Task CompilationPreservesTensorBytesArraysAndSourceIdentity()
    {
        using var fixture = new CompiledPackageFixture(arrays: true);
        var info = await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        using var source = GgufFile.Open(fixture.Source);
        using var packed = GgufFile.Open(fixture.Destination);

        await Assert.That(info.SourceSha256).IsEqualTo(Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(fixture.Source))));
        await Assert.That(packed.ReadStringArray("tokenizer.ggml.tokens").SequenceEqual(source.ReadStringArray("tokenizer.ggml.tokens"), StringComparer.Ordinal))
            .IsTrue().Because("Vocabulary entries must retain their token IDs.");
        await Assert.That(packed.ReadInt32Array("tokenizer.ggml.token_type").SequenceEqual(source.ReadInt32Array("tokenizer.ggml.token_type")))
            .IsTrue().Because("Token types must remain aligned with their vocabulary entries.");
        foreach (var tensor in source.Tensors.Values)
        {
            var compiled = packed.GetRequiredTensor(tensor.Name);
            await Assert.That(Bytes(packed, compiled).SequenceEqual(Bytes(source, tensor)))
                .IsTrue().Because($"Tensor '{tensor.Name}' must preserve every encoded byte at its original element position.");
            await Assert.That(compiled.SourceOffset).IsEqualTo(tensor.Offset);
            await Assert.That(compiled.Offset % 64).IsEqualTo(0L);
            await Assert.That(compiled.Type).IsEqualTo(tensor.Type);
        }
    }

    [Test]
    public async Task CompilationIsDeterministicAcrossDestinationNames()
    {
        using var fixture = new CompiledPackageFixture();
        var other = fixture.Destination + ".second.synapse";
        try
        {
            var first = await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
            var second = await CompiledPackageCompiler.CompileAsync(fixture.Source, other);
            await Assert.That(File.ReadAllBytes(other).SequenceEqual(File.ReadAllBytes(fixture.Destination)))
                .IsTrue().Because("Deterministic compilation must produce an identical byte sequence.");
            await Assert.That(first.Identity).IsEqualTo(second.Identity);
            await Assert.That(CompiledPackageReader.Inspect(other)).IsEqualTo(second);
        }
        finally
        {
            File.Delete(other);
        }
    }

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Native)]
    [Arguments(KernelBackend.Metal)]
    public async Task PackageExecutesDirectlyWithoutSourceAndMatchesEveryBackend(KernelBackend backend)
    {
        if (backend == KernelBackend.Metal)
        {
            _ = GpuHardware.RequireMetal();
        }

        // Head dimension 64 is supported by every existing executable backend.
        var source = TinyQwen2Gguf.Write(new(1, 1, 1, 64, 256, 64, 32), seed: 19);
        var destination = Path.ChangeExtension(source, ".synapse");
        try
        {
            var info = await CompiledPackageCompiler.CompileAsync(source, destination);
            var options = new ModelLoadOptions { ContextSize = 16, MaximumParallelism = 2, KernelBackend = backend };
            float[] logits;
            int[] generated;
            ModelGraphFingerprint graph;
            using (var original = (Qwen2Model)ModelLoader.LoadSourceForValidation(source, options))
            {
                logits = original.EvaluatePromptLogits([1, 2, 3]);
                generated = [.. original.Generate([1, 2, 3], 4).GeneratedTokens];
                graph = ModelGraphFingerprint.Compute(original.Graph);
            }

            File.Delete(source);
            using var model = (Qwen2Model)ModelLoader.Load(destination, options);
            await Assert.That(model.EvaluatePromptLogits([1, 2, 3]).Select(BitConverter.SingleToInt32Bits)
                .SequenceEqual(logits.Select(BitConverter.SingleToInt32Bits)))
                .IsTrue().Because($"The {backend} backend must preserve each vocabulary logit's FP32 bits.");
            await Assert.That(model.Generate([1, 2, 3], 4).GeneratedTokens.SequenceEqual(generated))
                .IsTrue().Because("Generated token order is part of the continuation.");
            await Assert.That(ModelGraphFingerprint.Compute(model.Graph)).IsEqualTo(graph);
            await Assert.That(info.GraphFingerprint).IsEqualTo(graph.Value);
        }
        finally
        {
            File.Delete(source);
            File.Delete(destination);
        }
    }

    [Test]
    [Arguments(1)]
    [Arguments(2)]
    public async Task PackagePreservesKQuantReferenceOutputs(int encoding)
    {
        using var fixture = new CompiledPackageFixture(encoding: (TinyEncoding)encoding);
        await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        var options = new ModelLoadOptions { ContextSize = 16, MaximumParallelism = 2, KernelBackend = KernelBackend.Reference };
        using var source = (Qwen2Model)ModelLoader.LoadSourceForValidation(fixture.Source, options);
        using var packed = (Qwen2Model)ModelLoader.Load(fixture.Destination, options);
        await Assert.That(packed.EvaluatePromptLogits([1, 2, 3]).Select(BitConverter.SingleToInt32Bits)
            .SequenceEqual(source.EvaluatePromptLogits([1, 2, 3]).Select(BitConverter.SingleToInt32Bits)))
            .IsTrue().Because("Lossless K-quant packing must preserve every vocabulary logit's FP32 bits.");
        await Assert.That(ModelGraphFingerprint.Compute(packed.Graph)).IsEqualTo(ModelGraphFingerprint.Compute(source.Graph));
    }

    private static unsafe byte[] Bytes(GgufFile file, GgufTensorInfo tensor) =>
        new ReadOnlySpan<byte>(file.GetTensorPointer(tensor), checked((int)tensor.ByteLength)).ToArray();
}
