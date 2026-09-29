using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.IntegrationTests.Features.GpuKernels;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

namespace ManagedCode.Synapse.IntegrationTests.Features.LayerDrop;

/// <summary>
/// Qualified layer drop (ADR-019): a dropped layer passes the residual through, costs no KV, and the model equals
/// the shallower model built from the kept layers.
/// </summary>
[NotInParallel]
public sealed class LayerDropTests
{
    private static readonly TinyQwen2Gguf.Shape Shape = new(Layers: 4, Heads: 7, KeyValueHeads: 1, HeadDimension: 64,
        FeedForward: 512, Vocabulary: 640, Context: 2_048);
    private static readonly int[] Prompt = [.. Enumerable.Range(0, 300).Select(index => index * 37 % 640)];
    private static readonly int[] Continuation = [5, 77, 300, 12];

    [Test]
    [Arguments(KernelBackend.Reference)]
    [Arguments(KernelBackend.Managed)]
    [Arguments(KernelBackend.Metal)]
    public async Task DroppedLayersEqualAShallowerModel(KernelBackend backend)
    {
        if (backend == KernelBackend.Metal)
        {
            GpuHardware.RequireMetal();
        }

        var full = TinyQwen2Gguf.Write(Shape, seed: 11);
        var shallow = TinyQwen2Gguf.Write(Shape, seed: 11, keptLayers: [0, 3]);
        try
        {
            using var dropped = Load(full, backend, new LayerDropProfile([1, 2]));
            using var reference = Load(shallow, backend, drop: null);

            var expected = reference.EvaluateIncrementalLogits(Prompt, Continuation);
            var actual = dropped.EvaluateIncrementalLogits(Prompt, Continuation);

            await Assert.That(actual).IsEquivalentTo(expected);
            await Assert.That(dropped.AllocatedKvBytes).IsEqualTo(reference.AllocatedKvBytes);
        }
        finally
        {
            File.Delete(full);
            File.Delete(shallow);
        }
    }

    [Test]
    public async Task LayerDropSizesKvForKeptLayers()
    {
        var path = TinyQwen2Gguf.Write(Shape, seed: 12);
        try
        {
            using var dense = Load(path, KernelBackend.Managed, drop: null);
            using var dropped = Load(path, KernelBackend.Managed, new LayerDropProfile([0, 2, 3]));

            _ = dense.EvaluatePromptLogits(Prompt);
            _ = dropped.EvaluatePromptLogits(Prompt);

            await Assert.That(dropped.AllocatedKvBytes * 4).IsEqualTo(dense.AllocatedKvBytes);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task LayerDropNeverPrefetchesDroppedWeights()
    {
        var path = TinyQwen2Gguf.Write(Shape, seed: 15);
        try
        {
            using var file = GgufFile.Open(path);
            var dense = Qwen2ModelComposition.PrefetchRanges(
                file, Qwen2ModelComposition.ReadDimensions(file, new ModelLoadOptions { ContextSize = 2_048 }));
            var dropped = Qwen2ModelComposition.PrefetchRanges(
                file, Qwen2ModelComposition.ReadDimensions(file, new ModelLoadOptions { ContextSize = 2_048, LayerDrop = new LayerDropProfile([1, 2]) }));
            var droppedBytes = file.Tensors.Values
                .Where(tensor => tensor.Name.StartsWith("blk.1.", StringComparison.Ordinal) || tensor.Name.StartsWith("blk.2.", StringComparison.Ordinal))
                .Sum(tensor => tensor.ByteLength);

            // Only alignment padding between tensors (at most 64 bytes each) may differ from the dropped tensors' bytes.
            var saved = dense.Sum(range => range.Length) - dropped.Sum(range => range.Length);
            await Assert.That(saved).IsGreaterThanOrEqualTo(droppedBytes).And.IsLessThanOrEqualTo(droppedBytes + (64 * 26));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task LayerDropIsNamedAndValidated()
    {
        var path = TinyQwen2Gguf.Write(Shape, seed: 13);
        try
        {
            var evidence = new string('a', 64);
            using var experimental = Load(path, KernelBackend.Managed, new LayerDropProfile([2, 1]));
            using var qualified = Load(path, KernelBackend.Managed, new LayerDropProfile([1]) { EvidenceSha256 = evidence });

            await Assert.That(experimental.RuntimeProfile).EndsWith("+drop2x");
            await Assert.That(qualified.RuntimeProfile).EndsWith("+drop1");
            foreach (var invalid in new[] { new LayerDropProfile([0, 1, 2, 3]), new LayerDropProfile([4]), new LayerDropProfile([1, 1]), new LayerDropProfile([]) })
            {
                await Assert.That(() => Load(path, KernelBackend.Managed, invalid)).Throws<ArgumentException>();
            }

            await Assert.That(() => Load(path, KernelBackend.Managed, new LayerDropProfile([1]) { EvidenceSha256 = "ABC" }))
                .Throws<ArgumentException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public async Task DroppedRegionsAreProfileBypassesInTheGraph()
    {
        var path = TinyQwen2Gguf.Write(Shape, seed: 14);
        try
        {
            using var dense = Load(path, KernelBackend.Managed, drop: null);
            using var dropped = Load(path, KernelBackend.Managed, new LayerDropProfile([1, 2]));
            RegionDescriptor Layer(Qwen2Model model, int layer)
            {
                return model.Graph.Regions.Single(region => region.SemanticAnnotations.Contains($"Layer:{layer}"));
            }

            await Assert.That(Layer(dropped, 1).Activation.Decision).IsTypeOf<ProfileDecision>();
            await Assert.That(Layer(dropped, 2).Activation.Provenance).IsTypeOf<ApproximateProvenance>();
            await Assert.That(Layer(dropped, 2).Activation.Skip).IsTypeOf<BypassOutputs>();
            await Assert.That(Layer(dropped, 0).Activation.Decision).IsTypeOf<AlwaysActive>();
            await Assert.That(Layer(dense, 1).Activation.Decision).IsTypeOf<AlwaysActive>();
            await Assert.That(ModelGraphFingerprint.Compute(dropped.Graph)).IsNotEqualTo(ModelGraphFingerprint.Compute(dense.Graph));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Qwen2Model Load(string path, KernelBackend backend, LayerDropProfile? drop) => (Qwen2Model)ModelLoader.Load(
        path,
        new ModelLoadOptions
        {
            ContextSize = 2_048,
            MaximumParallelism = 8,
            KernelBackend = backend,
            LayerDrop = drop,
        });
}
