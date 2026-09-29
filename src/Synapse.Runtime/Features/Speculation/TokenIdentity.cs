using System.Security.Cryptography;
using System.Text;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.Speculation;

/// <summary>
/// Token-ID identity between two models (ADR-020): a SHA-256 over the first <c>count</c> vocabulary entries, their
/// types, and every merge. A draft may propose tokens to a target only when both digest alike over the draft's
/// vocabulary. A file without tokenizer arrays digests as its (absent) vocabulary.
/// </summary>
internal static class TokenIdentity
{
    public static string Digest(GgufFile file, int count)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        if (file.Metadata.ContainsKey("tokenizer.ggml.tokens"))
        {
            var tokens = file.ReadStringArray("tokenizer.ggml.tokens");
            var types = file.ReadInt32Array("tokenizer.ggml.token_type");
            Append(hash, $"tokens:{Math.Min(count, tokens.Length)}");
            for (var index = 0; index < Math.Min(count, tokens.Length); index++)
            {
                Append(hash, tokens[index]);
                Append(hash, types[index].ToString(System.Globalization.CultureInfo.InvariantCulture));
            }

            foreach (var merge in file.ReadStringArray("tokenizer.ggml.merges"))
            {
                Append(hash, merge);
            }
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private static void Append(IncrementalHash hash, string value)
    {
        hash.AppendData(Encoding.UTF8.GetBytes(value));
        hash.AppendData([0]);
    }
}
