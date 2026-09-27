using System.Buffers.Binary;
using System.Security.Cryptography;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;

namespace ManagedCode.Synapse.IntegrationTests.Features.ModelPackages;

public sealed class SafeTensorHeaderTests
{
    private const string ModelSha256 =
        "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c";

    [Test]
    public async Task OfficialSmolLm2IndexIsBoundedAndComplete()
    {
        var path = GetFixturePath("model.safetensors");
        var index = SafeTensorHeaderReader.Read(path);
        await using var stream = File.OpenRead(path);
        var digest = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream));

        await Assert.That(digest).IsEqualTo(ModelSha256);
        await Assert.That(index.Tensors.Count).IsEqualTo(272);
        await Assert.That(index.Tensors.All(tensor =>
            tensor.DataType == SafeTensorDataType.BF16)).IsTrue();
        await Assert.That(index.Tensors.Sum(tensor => tensor.ElementCount)).IsEqualTo(134_515_008);
        await Assert.That(index.Tensors.Single(tensor =>
            tensor.Name == "model.embed_tokens.weight").Shape).IsEquivalentTo([49_152L, 576L]);
        await Assert.That(index.Tensors.Single(tensor =>
            tensor.Name == "model.layers.29.self_attn.k_proj.weight").Shape).IsEquivalentTo([192L, 576L]);
    }

    [Test]
    public async Task TruncatedHeaderFailsBeforePayloadAllocation()
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-{Guid.NewGuid():N}.safetensors");
        var bytes = new byte[16];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, 128);
        await File.WriteAllBytesAsync(path, bytes);
        try
        {
            await Assert.That(() => SafeTensorHeaderReader.Read(path))
                .Throws<InvalidDataException>();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public Task OfficialMiniLmEmbeddingIndexIsBounded() => AssertEmbeddingIndexAsync(
        "all-minilm-l6-v2-f32",
        "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        22_713_728);

    [Test]
    public Task OfficialBgeEmbeddingIndexIsBounded() => AssertEmbeddingIndexAsync(
        "bge-small-en-v1.5-f32",
        "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        33_360_512);

    private static async Task AssertEmbeddingIndexAsync(
        string packageId,
        string expectedDigest,
        long expectedParameters)
    {
        var path = GetPackageFile(packageId, "model.safetensors");
        var index = SafeTensorHeaderReader.Read(path);
        await using var stream = File.OpenRead(path);
        var digest = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream));

        await Assert.That(digest).IsEqualTo(expectedDigest);
        await Assert.That(index.Tensors.Sum(tensor => tensor.ElementCount)).IsEqualTo(expectedParameters);
        await Assert.That(index.Tensors.Any(tensor =>
            tensor.Name.EndsWith("embeddings.word_embeddings.weight", StringComparison.Ordinal)))
            .IsTrue();
    }

    private static string GetFixturePath(string fileName)
    {
        var configured = Environment.GetEnvironmentVariable("SYNAPSE_MODEL_ROOT");
        var modelRoot = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(FindRepositoryRoot(), "artifacts", "models")
            : Path.GetFullPath(configured);
        return Path.Combine(modelRoot, "smollm2-135m-instruct-bf16", fileName);
    }

    private static string GetPackageFile(string packageId, string fileName)
    {
        var configured = Environment.GetEnvironmentVariable("SYNAPSE_MODEL_ROOT");
        var modelRoot = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(FindRepositoryRoot(), "artifacts", "models")
            : Path.GetFullPath(configured);
        return Path.Combine(modelRoot, packageId, fileName);
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Synapse.slnx")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the Synapse repository root.");
    }
}
