using System.Globalization;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

internal static class OptimizationEvaluationCommand
{
    public static Task<int> RunAsync(string[] args) => OptimizationRunSupport.WithConsoleCancellationAsync(token => RunAsync(args, token));

    internal static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken)
    {
        if (!OptimizationRunSupport.Arguments(args, "--plan", out var input, out var output) || File.Exists(output))
        {
            await Console.Error.WriteLineAsync("Usage: optimization-eval --plan <json> --output <new.json>").ConfigureAwait(false);
            return 2;
        }

        var state = new OptimizationEvaluationState(output);
        try
        {
            state.PlanSha256 = await OptimizationValidation.FileSha256Async(input, cancellationToken).ConfigureAwait(false);
            state.Plan = await OptimizationValidation.ReadPlanAsync(input, cancellationToken).ConfigureAwait(false);
            var tokenizers = await PrepareAsync(state, cancellationToken).ConfigureAwait(false);
            var haystack = await File.ReadAllTextAsync(state.Plan.HaystackPath, cancellationToken).ConfigureAwait(false);
            state.HaystackSha256 = await OptimizationValidation.FileSha256Async(state.Plan.HaystackPath, cancellationToken).ConfigureAwait(false);
            var baseline = state.Plan.Profiles.Single(profile => profile.Id == state.Plan.BaselineProfile);
            var factory = new QualityTaskFactory(tokenizers[ModelPath(state.Plan, baseline)], haystack);
            var index = 0;
            foreach (var (context, quality) in Cases(state.Plan, factory))
            {
                ValidateCase(state.Plan, context, quality, tokenizers.Values);
                await RunCaseAsync(state, context, quality, index++, cancellationToken).ConfigureAwait(false);
            }

            state.Status = state.Samples.All(sample => sample.ExitCode == 0) ? "completed" : "failed";
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            state.Status = exception is OperationCanceledException ? "cancelled" : "failed";
            state.Error = exception.ToString();
            await Console.Error.WriteLineAsync(exception.Message).ConfigureAwait(false);
        }

        await state.WriteAsync().ConfigureAwait(false);
        return state.Status == "completed" ? 0 : state.Status == "cancelled" ? 130 : state.Samples.Count == 0 ? 2 : 1;
    }

    private static async Task<Dictionary<string, ITextTokenizer>> PrepareAsync(OptimizationEvaluationState state, CancellationToken cancellationToken)
    {
        var plan = state.Plan!;
        var tokenizers = new Dictionary<string, ITextTokenizer>(StringComparer.Ordinal);
        foreach (var path in plan.Profiles.Select(profile => ModelPath(plan, profile)).Distinct(StringComparer.Ordinal))
        {
            state.Packages.Add(await OptimizationValidation.DescribePackageAsync(path, cancellationToken).ConfigureAwait(false));
            cancellationToken.ThrowIfCancellationRequested();
            tokenizers.Add(path, TextTokenizers.FromModel(path));
        }

        var baseline = state.Packages.Single(package => package.Path == ModelPath(plan, plan.Profiles.Single(profile => profile.Id == plan.BaselineProfile)));
        if (state.Packages.Any(package => package.SourceSha256 != baseline.SourceSha256))
        {
            throw new InvalidDataException("All prepared candidates must share the dense baseline's original source SHA-256.");
        }

        return tokenizers;
    }

    private static IEnumerable<(OptimizationContext Context, QualityCase Quality)> Cases(OptimizationPlan plan, QualityTaskFactory factory)
    {
        for (var context = 0; context < plan.Contexts.Length; context++)
        {
            for (var task = 0; task < plan.Tasks.Length; task++)
            {
                var depths = plan.Tasks[task] == "vartrack" ? [0.5] : plan.Depths;
                for (var depth = 0; depth < depths.Length; depth++)
                {
                    var seed = unchecked(plan.Seed + (context * 1000) + (task * 100) + depth);
                    yield return (plan.Contexts[context], factory.Create(plan.Tasks[task], plan.Contexts[context].PromptTokens, depths[depth], seed));
                }
            }
        }
    }

    private static void ValidateCase(OptimizationPlan plan, OptimizationContext context, QualityCase quality, IEnumerable<ITextTokenizer> tokenizers)
    {
        if (quality.Tokens.Length < 2 || quality.Tokens.Length > context.PromptTokens ||
            (long)quality.Tokens.Length + plan.MaximumNewTokens > context.ContextSize)
        {
            throw new InvalidDataException("Generated quality prompt violates its target or prompt/output context bound.");
        }

        if (tokenizers.Any(tokenizer => !tokenizer.Encode(quality.Prompt, parseSpecialTokens: true).SequenceEqual(quality.Tokens)))
        {
            throw new InvalidDataException("Candidate tokenizer does not produce the identical prompt token IDs.");
        }
    }

    private static async Task RunCaseAsync(OptimizationEvaluationState state, OptimizationContext context, QualityCase quality, int caseIndex,
        CancellationToken cancellationToken)
    {
        var plan = state.Plan!;
        for (var round = 0; round < plan.Warmups + plan.Measurements; round++)
        {
            for (var index = 0; index < plan.Profiles.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var profile = plan.Profiles[(caseIndex + round + index) % plan.Profiles.Length];
                var path = ModelPath(plan, profile);
                var maximumNewTokens = Math.Min(quality.MaxTokens, plan.MaximumNewTokens);
                var request = new OptimizationRequest(1, path, context.ContextSize, quality.Tokens, [.. quality.Answers], maximumNewTokens,
                    plan.ScoredTailTokens, plan.Threads, plan.IncludeRepeatedPrompt, profile,
                    state.Packages.Single(package => package.Path == path), plan.RopeScaling);
                await Console.Error.WriteLineAsync(string.Create(CultureInfo.InvariantCulture,
                    $"optimization case={caseIndex} task={quality.Task} context={context.ContextSize} prompt={quality.Tokens.Length} depth={quality.Depth:F2} round={round} profile={profile.Id} starting"))
                    .ConfigureAwait(false);
                var child = await OptimizationChildProcess.RunAsync(request, plan.TimeoutSeconds, cancellationToken).ConfigureAwait(false);
                state.Samples.Add(new OptimizationSample(caseIndex, quality.Task, context.ContextSize, context.PromptTokens,
                    quality.Tokens.Length, maximumNewTokens, quality.Depth, quality.Seed, [.. quality.Answers], profile.Id, round, round < plan.Warmups,
                    child.ExitCode, child.WallMilliseconds, child.FootprintBytes, child.WorkingSetBytes, child.Result, child.Error));
                await state.WriteAsync().ConfigureAwait(false);
                await Console.Error.WriteLineAsync($"optimization case={caseIndex} round={round} profile={profile.Id} exit={child.ExitCode}").ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
            }
        }
    }

    private static string ModelPath(OptimizationPlan plan, OptimizationProfile profile) => profile.ModelPath ?? plan.ModelPath;
}
