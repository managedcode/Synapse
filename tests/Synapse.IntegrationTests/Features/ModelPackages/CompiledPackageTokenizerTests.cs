using ManagedCode.Synapse.Cli.Features.TextGeneration;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

[NotInParallel]
public sealed class CompiledPackageTokenizerTests
{
    [Test]
    public async Task LoadedTokenizerDecodesWithoutReopeningCompiledPath()
    {
        using var fixture = new CompiledPackageFixture(tokenizer: true);
        var sourceTokenizer = TextTokenizers.FromGguf(fixture.Source);
        await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        var moved = fixture.Destination + ".moved.synapse";
        try
        {
            using var model = ModelLoader.Load(fixture.Destination, new ModelLoadOptions { ContextSize = 16, MaximumParallelism = 1, KernelBackend = KernelBackend.Reference });
            File.Move(fixture.Destination, moved);
            var tokenizers = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Task.Run(model.CreateTokenizer)));
            var tokenizer = tokenizers[0];
            await Assert.That(GenerationCommand.DecodeOrNull(model, [0, 1])).IsEqualTo(sourceTokenizer.Decode([0, 1]));
            await Assert.That(tokenizer.Encode("ab").SequenceEqual(sourceTokenizer.Encode("ab")))
                .IsTrue().Because("Merged token IDs must retain their exact order.");
            await Assert.That(tokenizer.Encode("ba").SequenceEqual(sourceTokenizer.Encode("ba")))
                .IsTrue().Because("Unmerged token IDs must retain their exact order.");
            await Assert.That(tokenizer.VocabularySize).IsEqualTo(sourceTokenizer.VocabularySize);
            await Assert.That(tokenizers.All(instance => ReferenceEquals(instance, tokenizer))).IsTrue();
        }
        finally
        {
            File.Delete(moved);
        }
    }

    [Test]
    public async Task UnsupportedLoadedTokenizerStillReturnsNullCliText()
    {
        using var fixture = new CompiledPackageFixture();
        await CompiledPackageCompiler.CompileAsync(fixture.Source, fixture.Destination);
        using var model = ModelLoader.Load(fixture.Destination, new ModelLoadOptions { ContextSize = 16, MaximumParallelism = 1, KernelBackend = KernelBackend.Reference });
        await Assert.That(() => model.CreateTokenizer()).Throws<NotSupportedException>();
        await Assert.That(GenerationCommand.DecodeOrNull(model, [1, 2])).IsNull();
    }
}
