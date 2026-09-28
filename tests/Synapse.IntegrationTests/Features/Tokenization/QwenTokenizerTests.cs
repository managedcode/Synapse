using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.Tokenization;

/// <summary>
/// The repo-owned byte-level BPE tokenizer reads its vocabulary and merges from the GGUF file. Expected IDs come
/// from the pinned llama.cpp `llama-tokenize` (b29c606e2) over the same model file.
/// </summary>
public sealed class QwenTokenizerTests
{
    private static readonly (string Text, int[] Ids)[] References =
    [
        ("The capital of France is", [785, 6722, 315, 9625, 374]),
        (" Paris", [12095]),
        ("Hello, world! It's 2026-09-28.", [9707, 11, 1879, 0, 1084, 594, 220, 17, 15, 17, 21, 12, 15, 24, 12, 17, 23, 13]),
        ("Київ — столиця України.", [26338, 1802, 144031, 5474, 1959, 130297, 1802, 10373, 4235, 127872, 144031, 22621, 13]),
        ("café 🙂 naïve", [924, 58858, 27484, 94880, 586]),
        ("fn main() {\n    println!(\"hi\");\n}\n", [8822, 1887, 368, 341, 262, 13751, 17223, 6023, 797, 532]),
        ("12345 67890", [16, 17, 18, 19, 20, 220, 21, 22, 23, 24, 15]),
        ("a  b\t\tc\n\n\nd   ", [64, 220, 293, 197, 1444, 1406, 67, 262]),
        ("I'll don't WE'RE they've", [40, 3278, 1513, 944, 19677, 94153, 807, 3003]),
        ("   leading spaces and trailing  ", [256, 6388, 12621, 323, 27748, 256]),
        ("日本語のテキスト", [101059, 102819, 15767, 56833, 61803, 70534]),
    ];

    [Test]
    public async Task EncodeMatchesLlamaCppReference()
    {
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());

        foreach (var (text, ids) in References)
        {
            await Assert.That(tokenizer.Encode(text, parseSpecialTokens: false)).IsEquivalentTo(ids).Because(text);
        }
    }

    [Test]
    public async Task DecodeRoundTripsText()
    {
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());

        foreach (var (text, ids) in References)
        {
            await Assert.That(tokenizer.Decode(ids)).IsEqualTo(text);
        }
    }

    [Test]
    public async Task SpecialTokensAreParsedAndRenderedOnlyWhenAsked()
    {
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
        const string Chat = "<|im_start|>user\nhi<|im_end|>";

        var parsed = tokenizer.Encode(Chat, parseSpecialTokens: true);
        var literal = tokenizer.Encode(Chat, parseSpecialTokens: false);

        await Assert.That(parsed).IsEquivalentTo([151644, 872, 198, 6023, 151645]);
        await Assert.That(literal).IsEquivalentTo([27, 91, 318, 4906, 91, 29, 872, 198, 6023, 27, 91, 318, 6213, 91, 29]);
        await Assert.That(tokenizer.Decode(parsed, includeSpecialTokens: true)).IsEqualTo(Chat);
        await Assert.That(tokenizer.Decode(parsed)).IsEqualTo("user\nhi");
    }

    [Test]
    public async Task ChatTemplateReproducesPasskeyFixture()
    {
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
        using var scenario = JsonDocument.Parse(await File.ReadAllTextAsync(ScenarioPath("passkey-qwen2.5.json")));
        var segments = scenario.RootElement.GetProperty("segments");
        int[] Ids(string name)
        {
            return [.. segments.GetProperty(name).EnumerateArray().Select(value => value.GetInt32())];
        }

        var prompt = ChatTemplates.Qwen(
            "You are a helpful assistant.",
            [new ChatMessage("user", "There is an important pass key hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the pass key later.")],
            addGenerationPrompt: false);

        var prefix = tokenizer.Encode(prompt[..prompt.LastIndexOf("<|im_end|>", StringComparison.Ordinal)]);

        await Assert.That(prefix).IsEquivalentTo(Ids("prefix"));
        await Assert.That(tokenizer.Encode(" The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."))
            .IsEquivalentTo(Ids("filler"));
    }

    [Test]
    public async Task UnsupportedPreTokenizerFailsExplicitly()
    {
        var exception = await Assert.That(() => TextTokenizers.RequireSupported("gpt2", "llama-bpe"))
            .Throws<NotSupportedException>();

        await Assert.That(exception!.Message).Contains("llama-bpe");
        await Assert.That(() => TextTokenizers.RequireSupported("gpt2", "qwen2")).ThrowsNothing();
    }

    private static string ScenarioPath(string name)
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "Synapse.slnx")))
        {
            directory = directory.Parent;
        }

        return Path.Combine(directory!.FullName, "benchmarks", "scenarios", name);
    }
}
