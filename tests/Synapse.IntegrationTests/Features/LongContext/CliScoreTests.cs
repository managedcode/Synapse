using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.IntegrationTests.Features.LongContext;

/// <summary><c>synapse score</c> (ADR-015) and <c>synapse tokenize/detokenize</c> (ADR-014) as real processes.</summary>
[NotInParallel]
public sealed class CliScoreTests
{
    [Test]
    public async Task CliScoreReportsPerplexityWithTheLlamaProtocol()
    {
        var text = string.Concat(Enumerable.Repeat(TokenScoringTests.Text + "\n", 4));
        var ids = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath()).Encode(text, parseSpecialTokens: false);
        var textFile = WriteTemporary(text);
        var traceFile = textFile + ".trace.json";
        try
        {
            var (exit, output, error) = await RunCliAsync(
                "score", "--text-file", textFile, "--context-size", "64", "--chunks", "2",
                "--backend", "managed", "--scoring-rows", "8", "--scores-output", traceFile);

            await Assert.That(exit).IsEqualTo(0).Because(error);
            using var json = JsonDocument.Parse(output);
            var root = json.RootElement;
            await Assert.That(root.GetProperty("token_count").GetInt32()).IsEqualTo(ids.Count);
            await Assert.That(root.GetProperty("tokens_sha256").GetString()).IsEqualTo(Sha256(ids));
            await Assert.That(root.GetProperty("chunks").GetInt32()).IsEqualTo(2);
            await Assert.That(root.GetProperty("first_scored_position").GetInt32()).IsEqualTo(32);
            await Assert.That(root.GetProperty("scored_tokens").GetInt32()).IsEqualTo(2 * 31);
            await Assert.That(root.GetProperty("cumulative_perplexity").GetArrayLength()).IsEqualTo(2);
            await Assert.That(root.GetProperty("perplexity").GetDouble()).IsGreaterThan(1).And.IsLessThan(100);
            await Assert.That(error).Contains("score: chunk 2/2 ppl");
            using var trace = JsonDocument.Parse(await File.ReadAllTextAsync(traceFile));
            var chunks = trace.RootElement.GetProperty("chunks");
            await Assert.That(chunks.GetArrayLength()).IsEqualTo(2);
            await Assert.That(chunks[1].GetProperty("negative_log_likelihoods").GetArrayLength()).IsEqualTo(31);
        }
        finally
        {
            File.Delete(textFile);
            File.Delete(traceFile);
        }
    }

    [Test]
    public async Task CliScoreCanScoreOnlyTheTailOfEachChunk()
    {
        var textFile = WriteTemporary(string.Concat(Enumerable.Repeat(TokenScoringTests.Text + "\n", 4)));
        try
        {
            var (exit, output, error) = await RunCliAsync(
                "score", "--text-file", textFile, "--context-size", "64", "--chunks", "2", "--first-scored", "60",
                "--backend", "managed", "--scoring-rows", "1");
            var (badExit, _, _) = await RunCliAsync(
                "score", "--text-file", textFile, "--context-size", "64", "--first-scored", "63");

            await Assert.That(exit).IsEqualTo(0).Because(error);
            using var json = JsonDocument.Parse(output);
            await Assert.That(json.RootElement.GetProperty("first_scored_position").GetInt32()).IsEqualTo(60);
            await Assert.That(json.RootElement.GetProperty("scored_tokens").GetInt32()).IsEqualTo(2 * 3);
            await Assert.That(badExit).IsEqualTo(2);
        }
        finally
        {
            File.Delete(textFile);
        }
    }

    [Test]
    public async Task CliScoreRejectsInputShorterThanOneChunk()
    {
        var textFile = WriteTemporary("Too short.");
        try
        {
            var (exit, _, error) = await RunCliAsync("score", "--text-file", textFile, "--context-size", "64");
            var (usageExit, _, usage) = await RunCliAsync("score", "--text-file", textFile);

            await Assert.That(exit).IsEqualTo(1);
            await Assert.That(error).Contains("at least 64 tokens");
            await Assert.That(usageExit).IsEqualTo(2);
            await Assert.That(usage).Contains("--context-size <n>");
        }
        finally
        {
            File.Delete(textFile);
        }
    }

    [Test]
    public async Task CliTokenizeAndDetokenizeRoundTrip()
    {
        const string Text = "<|im_start|>user\nКиїв — столиця України. printf(\"%s\\n\");<|im_end|>";
        var tokenizer = TextTokenizers.FromGguf(ReferenceBenchmarkFixture.GetModelPath());
        var textFile = WriteTemporary(Text);
        var idsFile = textFile + ".ids";
        try
        {
            var (exit, ids, error) = await RunCliAsync("tokenize", "--text-file", textFile);
            await File.WriteAllTextAsync(idsFile, ids);
            var (decodeExit, decoded, _) = await RunCliAsync("detokenize", "--tokens-file", idsFile, "--special");
            var (literalExit, literal, _) = await RunCliAsync("tokenize", "--text-file", textFile, "--no-parse-special");

            await Assert.That(exit).IsEqualTo(0).Because(error);
            await Assert.That(ParseIds(ids)).IsEquivalentTo(tokenizer.Encode(Text));
            await Assert.That(decodeExit).IsEqualTo(0);
            await Assert.That(decoded).IsEqualTo(Text);
            await Assert.That(literalExit).IsEqualTo(0);
            await Assert.That(ParseIds(literal)).IsEquivalentTo(tokenizer.Encode(Text, parseSpecialTokens: false));
        }
        finally
        {
            File.Delete(textFile);
            File.Delete(idsFile);
        }
    }

    [Test]
    public async Task CliDetokenizeRejectsIdsOutsideTheVocabulary()
    {
        var idsFile = WriteTemporary("785\n999999\n");
        try
        {
            var (exit, _, error) = await RunCliAsync("detokenize", "--tokens-file", idsFile);

            await Assert.That(exit).IsEqualTo(1);
            await Assert.That(error).Contains("outside the vocabulary");
        }
        finally
        {
            File.Delete(idsFile);
        }
    }

    [Test]
    public async Task CliGenerateReportsDecodedText()
    {
        var (exit, output, error) = await RunCliAsync(
            "generate", "--tokens", "785,6722,315,9625,374", "--max-tokens", "2", "--context-size", "64",
            "--backend", "managed", "--threads", "2");

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        var root = json.RootElement;
        await Assert.That(root.GetProperty("generated_tokens").EnumerateArray().Select(id => id.GetInt32()))
            .IsEquivalentTo([12095, 13]);
        await Assert.That(root.GetProperty("generated_text").GetString()).IsEqualTo(" Paris.");
    }

    [Test]
    public async Task CliKvPagesReachTheRuntimeProfileAndRejectMalformedValues()
    {
        string[] generate = ["generate", "--tokens", "785,6722,315,9625,374", "--max-tokens", "1", "--context-size", "64",
            "--backend", "managed", "--threads", "2"];

        var (exit, output, error) = await RunCliAsync(generate[0], [.. generate[1..], "--kv-pages", "2:64:p16:random:7"]);

        await Assert.That(exit).IsEqualTo(0).Because(error);
        using var json = JsonDocument.Parse(output);
        await Assert.That(json.RootElement.GetProperty("subject").GetString()).EndsWith("+kvpages2w64p16r");
        foreach (var malformed in new[] { "2", "2:64:p24", "2:64:random:x", "2:64:extra", "2:64:7" })
        {
            var (badExit, _, _) = await RunCliAsync(generate[0], [.. generate[1..], "--kv-pages", malformed]);
            await Assert.That(badExit).IsEqualTo(2).Because(malformed);
        }
    }

    internal static string Sha256(IReadOnlyList<int> ids) => Convert.ToHexStringLower(SHA256.HashData(
        Encoding.UTF8.GetBytes(string.Concat(ids.Select(id => id.ToString(CultureInfo.InvariantCulture) + "\n")))));

    private static int[] ParseIds(string text) =>
        [.. text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(value => int.Parse(value, CultureInfo.InvariantCulture))];

    private static string WriteTemporary(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"synapse-score-{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, content);
        return path;
    }

    internal static async Task<(int ExitCode, string Output, string Error)> RunCliAsync(string command, params string[] extra)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        startInfo.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "synapse.dll"));
        startInfo.ArgumentList.Add(command);
        startInfo.ArgumentList.Add("--model");
        startInfo.ArgumentList.Add(ReferenceBenchmarkFixture.GetModelPath());
        foreach (var argument in extra)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = Process.Start(startInfo) ?? throw new InvalidOperationException("The CLI did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        return (process.ExitCode, await output, await error);
    }
}
