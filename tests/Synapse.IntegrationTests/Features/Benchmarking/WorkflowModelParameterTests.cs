using System.Text;
using static ManagedCode.Synapse.IntegrationTests.Features.Benchmarking.ReferenceBenchmarkFixture;

namespace ManagedCode.Synapse.IntegrationTests.Features.Benchmarking;

// TEST-BMK-001-5: workflow model selection propagates through the real repository-owned pipeline contract.
public sealed class WorkflowModelParameterTests
{
    [Test]
    public async Task CpuModelInputsDriveAcquisitionPreparationMeasurementsAndCache()
    {
        var workflow = await ReadWorkflowAsync("performance");
        await AssertParameterAsync(workflow, "gguf_model_id", "SYNAPSE_GGUF_MODEL_ID");
        await AssertParameterAsync(workflow, "gguf_model_file", "SYNAPSE_GGUF_MODEL_FILE");
        var scripts = string.Join('\n', RunBodies(workflow));
        await Assert.That(scripts).Contains("model fetch --id \"${SYNAPSE_GGUF_MODEL_ID}\"");
        await Assert.That(scripts).Contains("${SYNAPSE_MODEL_ROOT}/${SYNAPSE_GGUF_MODEL_ID}/${SYNAPSE_GGUF_MODEL_FILE}");
        await AssertPreparationIfPresentAsync(scripts);
        await Assert.That(scripts.Split("--model \"${SYNAPSE_MODEL_ROOT}/${SYNAPSE_GGUF_MODEL_ID}/${SYNAPSE_GGUF_MODEL_FILE}\"",
            StringSplitOptions.None).Length).IsGreaterThanOrEqualTo(3)
            .Because("both the smoke matrix and longer diagnostics consume the selected source");
        await Assert.That(workflow).Contains("env.SYNAPSE_GGUF_MODEL_ID");
        await Assert.That(workflow).Contains("env.SYNAPSE_GGUF_MODEL_FILE");
        await AssertNoModelLiteralsAsync(scripts);
    }

    [Test]
    public async Task SmokeQualificationInputsTravelTogetherWithTheSelectedModel()
    {
        var workflow = await ReadWorkflowAsync("performance");
        var parameters = new (string Input, string Environment, string Argument)[]
        {
            ("smoke_prompt", "SYNAPSE_SMOKE_PROMPT", "--prompt"),
            ("smoke_prompt_token_ids", "SYNAPSE_SMOKE_PROMPT_TOKEN_IDS", "--prompt-token-ids"),
            ("smoke_expected_token_ids", "SYNAPSE_SMOKE_EXPECTED_TOKEN_IDS", "--expected-token-ids"),
            ("smoke_expected_text", "SYNAPSE_SMOKE_EXPECTED_TEXT", "--expected-text"),
            ("smoke_max_tokens", "SYNAPSE_SMOKE_MAX_TOKENS", "--max-tokens"),
        };
        var scripts = string.Join('\n', RunBodies(workflow));
        foreach (var (input, environment, argument) in parameters)
        {
            await AssertParameterAsync(workflow, input, environment);
            await Assert.That(scripts).Contains($"{argument} \"${{{environment}}}\"");
        }

        await Assert.That(scripts).DoesNotContain("785,6722,315,9625,374");
        await Assert.That(scripts).DoesNotContain("12095,13,1084,374,279,7772,3283,304");
        await Assert.That(scripts).DoesNotContain("The capital of France is");
    }

    [Test]
    public async Task MlxAndFoundrySelectionsReachAcquisitionExecutionAndAggregation()
    {
        var workflow = await ReadWorkflowAsync("performance");
        await AssertParameterAsync(workflow, "mlx_model_id", "SYNAPSE_MLX_MODEL_ID");
        await AssertParameterAsync(workflow, "foundry_model_set", "SYNAPSE_FOUNDRY_MODEL_SET");
        var scripts = string.Join('\n', RunBodies(workflow));
        await Assert.That(scripts).Contains("model fetch --id \"${SYNAPSE_MLX_MODEL_ID}\"");
        await Assert.That(scripts).Contains("--model \"${SYNAPSE_MODEL_ROOT}/${SYNAPSE_MLX_MODEL_ID}\"");
        await Assert.That(scripts.Split("--set \"${SYNAPSE_FOUNDRY_MODEL_SET}\"", StringSplitOptions.None).Length)
            .IsGreaterThanOrEqualTo(4).Because("Foundry plan, fetch and run use the same selected model set");
        await Assert.That(scripts).Contains("--model-set \"${SYNAPSE_FOUNDRY_MODEL_SET}\"");
        await Assert.That(workflow).Contains("env.SYNAPSE_MLX_MODEL_ID");
        await AssertNoModelLiteralsAsync(scripts);
    }

    [Test]
    public async Task VerificationSelectionsReachFetchPreparationFixturesAndCaches()
    {
        var workflow = await ReadWorkflowAsync("verify");
        var parameters = new (string Input, string Environment)[]
        {
            ("gguf_model_id", "SYNAPSE_GGUF_MODEL_ID"),
            ("gguf_model_file", "SYNAPSE_GGUF_MODEL_FILE"),
            ("model_set", "SYNAPSE_MODEL_SET"),
            ("embedding_model_set", "SYNAPSE_EMBEDDING_MODEL_SET"),
            ("foundry_model_set", "SYNAPSE_FOUNDRY_MODEL_SET"),
            ("foundry_anchor", "SYNAPSE_FOUNDRY_ANCHOR"),
        };
        foreach (var (input, environment) in parameters)
        {
            await AssertParameterAsync(workflow, input, environment);
        }

        var scripts = string.Join('\n', RunBodies(workflow));
        await Assert.That(scripts).Contains("model fetch --id \"${SYNAPSE_GGUF_MODEL_ID}\"");
        await Assert.That(scripts).Contains("model fetch --set \"${SYNAPSE_MODEL_SET}\"");
        await Assert.That(scripts).Contains("model fetch --set \"${SYNAPSE_EMBEDDING_MODEL_SET}\"");
        await AssertPreparationIfPresentAsync(scripts);
        await Assert.That(scripts).Contains("--set \"${SYNAPSE_FOUNDRY_MODEL_SET}\" --alias \"${SYNAPSE_FOUNDRY_ANCHOR}\"");
        foreach (var selection in new[] { "SYNAPSE_GGUF_MODEL_ID", "SYNAPSE_GGUF_MODEL_FILE", "SYNAPSE_MODEL_SET",
            "SYNAPSE_EMBEDDING_MODEL_SET", "SYNAPSE_FOUNDRY_ANCHOR", "SYNAPSE_FOUNDRY_MODEL_SET" })
        {
            await Assert.That(workflow).Contains($"env.{selection}");
        }

        await AssertNoModelLiteralsAsync(scripts);
    }

    private static Task<string> ReadWorkflowAsync(string name) =>
        File.ReadAllTextAsync(Path.Combine(FindRepositoryRoot(), ".github", "workflows", name + ".yml"));

    private static async Task AssertParameterAsync(string workflow, string input, string environment)
    {
        await Assert.That(workflow.Contains($"      {input}:", StringComparison.Ordinal)).IsTrue()
            .Because($"workflow_dispatch declares {input}");
        var configuration = workflow.Split('\n').SingleOrDefault(line => line.TrimStart()
            .StartsWith(environment + ":", StringComparison.Ordinal)) ?? string.Empty;
        await Assert.That(configuration).Contains($"inputs.{input} || vars.{environment}")
            .Because("manual selection and repository variables share the declared default configuration");
    }

    private static async Task AssertPreparationIfPresentAsync(string scripts)
    {
        if (scripts.Contains("model compile", StringComparison.Ordinal))
        {
            await Assert.That(scripts).Contains("${SYNAPSE_MODEL_ROOT}/${SYNAPSE_GGUF_MODEL_ID}/${SYNAPSE_GGUF_MODEL_FILE}");
            await Assert.That(scripts).Contains("model compile --source \"${source_model}\"");
        }
    }

    private static async Task AssertNoModelLiteralsAsync(string scripts)
    {
        foreach (var literal in new[] { "qwen2.5-0.5b", "qwen3-0.6b", "phi-3.5-mini", "phi-4-mini",
            "mistral-7b", "deepseek-r1", "foundry-local-families.json", "--set family-small", "--set embedding-small" })
        {
            await Assert.That(scripts).DoesNotContain(literal)
                .Because("model-specific defaults belong to configuration rather than command bodies");
        }
    }

    private static IEnumerable<string> RunBodies(string workflow)
    {
        var lines = workflow.Split('\n');
        for (var index = 0; index < lines.Length; index++)
        {
            var content = lines[index].TrimStart();
            if (!content.StartsWith("run:", StringComparison.Ordinal))
            {
                continue;
            }

            var indentation = lines[index].Length - content.Length;
            var body = new StringBuilder(content[4..]);
            while (index + 1 < lines.Length && (string.IsNullOrWhiteSpace(lines[index + 1]) ||
                lines[index + 1].TakeWhile(char.IsWhiteSpace).Count() > indentation))
            {
                _ = body.AppendLine().Append(lines[++index]);
            }

            yield return body.ToString();
        }
    }
}
