using System.Text.Json;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

[NotInParallel]
public sealed class NativeLlamaCppSmokeTests
{
    [Test]
    public async Task MatchesPromptAndContinuationWithoutInventedTimings()
    {
        var executable = RequireEnvironmentFile("SYNAPSE_LLAMACPP_EXECUTABLE");
        var version = Environment.GetEnvironmentVariable("SYNAPSE_LLAMACPP_VERSION")
            ?? throw new InvalidOperationException("SYNAPSE_LLAMACPP_VERSION must identify the tested binary.");
        using var result = await RunSubjectAsync(
            "llamacpp",
            "--subject-executable", executable,
            "--subject-version", version,
            "--expected-prompt-token-ids", string.Join(',', PromptTokens));

        var root = result.RootElement;
        await Assert.That(root.GetProperty("subject").GetString()).IsEqualTo("llamacpp");
        await Assert.That(root.GetProperty("subject_wall_milliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("subject_version").GetString()).IsEqualTo(version);
        await Assert.That(root.GetProperty("generated_tokens").GetInt32()).IsEqualTo(8);
        await Assert.That(root.GetProperty("text").GetString()).IsEqualTo(" Paris. It is the largest city in");
        await Assert.That(root.GetProperty("prompt_token_ids").EnumerateArray().Select(token => token.GetInt32()))
            .IsEquivalentTo(PromptTokens);
        await Assert.That(root.GetProperty("load_milliseconds").ValueKind).IsEqualTo(JsonValueKind.Null);
        await Assert.That(root.GetProperty("time_to_first_token_milliseconds").ValueKind)
            .IsEqualTo(JsonValueKind.Null);
        await Assert.That(root.GetProperty("decode_tokens_per_second").ValueKind)
            .IsEqualTo(JsonValueKind.Null);
        await Assert.That(root.GetProperty("native_prompt_eval_milliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("native_eval_milliseconds").GetDouble()).IsGreaterThan(0);
        await Assert.That(root.GetProperty("measurement_scope").GetString())
            .IsEqualTo("native_internal_plus_process_observed");
    }

    [Test]
    public async Task RejectsDifferentPromptTokens()
    {
        var executable = RequireEnvironmentFile("SYNAPSE_LLAMACPP_EXECUTABLE");
        var result = await RunSubjectProcessAsync(
            "llamacpp",
            "--subject-executable", executable,
            "--subject-version", "b29c606e2",
            "--expected-prompt-token-ids", "1,2,3,4,5");

        await Assert.That(result.ExitCode).IsEqualTo(1);
        await Assert.That(result.StandardOutput).IsEmpty();
        await Assert.That(result.StandardError).Contains("prompt token IDs differ");
    }

    [Test]
    public async Task SingleTokenHasNoDecodeRate()
    {
        var executable = RequireEnvironmentFile("SYNAPSE_LLAMACPP_EXECUTABLE");
        using var result = await RunSubjectAsync(
            "llamacpp",
            "--subject-executable", executable,
            "--subject-version", "b29c606e2",
            "--expected-prompt-token-ids", string.Join(',', PromptTokens),
            "--max-tokens", "1");

        var root = result.RootElement;
        await Assert.That(root.GetProperty("generated_tokens").GetInt32()).IsEqualTo(1);
        await Assert.That(root.GetProperty("text").GetString()).IsEqualTo(" Paris");
        await Assert.That(root.GetProperty("native_eval_tokens_per_second").ValueKind)
            .IsEqualTo(JsonValueKind.Null);
    }
}
