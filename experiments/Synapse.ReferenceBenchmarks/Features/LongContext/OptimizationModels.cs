using System.Text.Json.Serialization;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;

internal sealed record OptimizationPlan
{
    public int SchemaVersion { get; set; } = 1;
    public string ModelPath { get; set; } = string.Empty;
    public string HaystackPath { get; set; } = string.Empty;
    public string BaselineProfile { get; set; } = "dense";
    public OptimizationContext[] Contexts { get; set; } = [];
    public double[] Depths { get; set; } = [0.1, 0.5, 0.9];
    public string[] Tasks { get; set; } = ["needle", "multikey", "vartrack"];
    public int Seed { get; set; } = 20261003;
    public int MaximumNewTokens { get; set; } = 32;
    public int ScoredTailTokens { get; set; } = 32;
    public int Warmups { get; set; } = 1;
    public int Measurements { get; set; } = 3;
    public int Threads { get; set; } = 8;
    public int TimeoutSeconds { get; set; } = 900;
    public bool IncludeRepeatedPrompt { get; set; } = true;
    public RopeScaling? RopeScaling { get; set; }
    public OptimizationProfile[] Profiles { get; set; } = [];
}

internal sealed record OptimizationContext(int ContextSize, int PromptTokens);

internal sealed record OptimizationProfile
{
    public string Id { get; set; } = string.Empty;
    public string? ModelPath { get; set; }
    public string Backend { get; set; } = "managed";
    public string KvPrecision { get; set; } = "f32";
    public bool ReusePromptPrefix { get; set; }
    public OptimizationKvPages? KvPages { get; set; }
    public int[]? DropLayers { get; set; }
    public string? EvidenceSha256 { get; set; }
}

internal sealed record OptimizationKvPages
{
    public int BudgetPages { get; set; }
    public int WindowTokens { get; set; }
    public int PageTokens { get; set; } = 64;
    public KvPageSelection Selection { get; set; } = KvPageSelection.KeyBound;
    public int Seed { get; set; }

    internal KvPageActivation ToRuntime() => new(BudgetPages, WindowTokens)
    {
        PageTokens = PageTokens,
        Selection = Selection,
        Seed = Seed,
    };
}

internal sealed record OptimizationPackage(
    string Path, string FileSha256, string Identity, string SourceSha256,
    string GraphFingerprint);

internal sealed record OptimizationRequest(
    int SchemaVersion, string ModelPath, int ContextSize, int[] PromptTokens, string[] Answers,
    int MaximumNewTokens, int ScoredTailTokens, int Threads, bool IncludeRepeatedPrompt,
    OptimizationProfile Profile, OptimizationPackage? Package = null, RopeScaling? RopeScaling = null);

internal sealed record OptimizationGeneration(
    string Turn, int PromptTokens, int MaximumNewTokens, int[] GeneratedTokens, string Output, int ReusedPromptTokens,
    double? PrefillMilliseconds, double TimeToFirstTokenMilliseconds, double GenerationMilliseconds, double? DecodeTokensPerSecond,
    double AnswerScore, bool ContainsAllAnswers, long AllocatedKvBytes);

internal sealed record OptimizationScore(
    int FirstScoredPosition, double[] NegativeLogLikelihoods, int[] GreedyTokens,
    double MeanNegativeLogLikelihood, double LogPerplexity, double? TailPerplexity,
    string PerplexityStatus, double ScoringMilliseconds);

internal sealed record OptimizationWorkerResult(
    string Status, string? Error, int ProcessId, string? PackageIdentity, string? PackageFileSha256,
    string? SourceSha256, string PromptTokenIdsSha256, string? GraphFingerprint,
    string? RuntimeProfile, string? KernelImplementation, double? LoadMilliseconds,
    OptimizationGeneration[] Generations, OptimizationScore? Score);

internal sealed record OptimizationSample(
    int CaseIndex, string Task, int ContextSize, int TargetPromptTokens, int PromptTokens, int MaximumNewTokens,
    double Depth, int Seed, string[] Answers, string ProfileId, int Round, bool Warmup,
    int ExitCode, double ProcessWallMilliseconds, long? PeakPhysicalFootprintBytes,
    long? PeakWorkingSetBytes, OptimizationWorkerResult? Result, string? Error);

internal sealed record OptimizationComparison(
    int CaseIndex, int Round, bool Warmup, string BaselineProfile, string CandidateProfile,
    bool TokenIdsIdentical, double? MeanAbsoluteNegativeLogLikelihoodDelta,
    double? GreedyAgreement, double? TailPerplexityRatio, string? TailPerplexityRatioStatus, OptimizationTurnComparison[] Turns);

internal sealed record OptimizationTurnComparison(
    string Turn, int BaselineGeneratedTokens, int CandidateGeneratedTokens,
    bool GeneratedTokensIdentical, double PositionalTokenAgreement,
    bool BaselineAnswerCorrect, bool CandidateAnswerCorrect,
    double TimeToFirstTokenSpeedup, double GenerationSpeedup);

internal sealed record OptimizationEvidence(
    int SchemaVersion, string Kind, string Status, DateTimeOffset RecordedAtUtc,
    string OperatingSystem, string Architecture, string Runtime, string PlanSha256,
    string? HaystackSha256, OptimizationPlan? Plan, OptimizationPackage[] Packages,
    OptimizationSample[] Samples, OptimizationComparison[] Comparisons, string[] Limitations, string? Error);

[JsonSerializable(typeof(OptimizationPlan))]
[JsonSerializable(typeof(OptimizationRequest))]
[JsonSerializable(typeof(OptimizationWorkerResult))]
[JsonSerializable(typeof(OptimizationEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, UseStringEnumConverter = true)]
internal sealed partial class OptimizationJsonContext : JsonSerializerContext;
