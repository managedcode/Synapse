using System.Text.Json.Serialization;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;

internal sealed record MemoryEvaluationRequest
{
    public int SchemaVersion { get; set; } = 1;
    public string ModelPath { get; set; } = string.Empty;
    public int ContextSize { get; set; }
    public int[] ShortPromptTokens { get; set; } = [];
    public int[] LongPromptTokens { get; set; } = [];
    public int MaximumNewTokens { get; set; } = 16;
    public int ScoredTailTokens { get; set; }
    public int Threads { get; set; } = 8;
    public OptimizationProfile Profile { get; set; } = new() { Id = "memory" };
    public RopeScaling? RopeScaling { get; set; }
}

internal sealed record MemoryGeneration(
    string PromptTokenIdsSha256, int PromptTokens, int MaximumNewTokens, int[] GeneratedTokens,
    int ReusedPromptTokens, double? PrefillMilliseconds, double TimeToFirstTokenMilliseconds,
    double GenerationMilliseconds);

internal sealed record MemoryPhase(
    string Name, long AllocatedKvBytes, long WorkingSetBytes, long? PhysicalFootprintBytes,
    long ManagedHeapBytes, double PhaseMilliseconds, string RuntimeProfile, string KernelImplementation,
    string GraphFingerprint, MemoryGeneration? Generation = null, OptimizationScore? Score = null);

internal sealed record MemoryQualityComparison(
    string Phase, string FreshPhase, bool PromptTokenIdsIdentical, bool GeneratedTokensIdentical,
    double? MeanAbsoluteNegativeLogLikelihoodDelta, double? GreedyAgreement,
    double? TailPerplexityRatio, string? TailPerplexityRatioStatus);

internal sealed record MemoryEvaluationEvidence(
    int SchemaVersion, string Kind, string Status, string? Error, DateTimeOffset RecordedAtUtc,
    string OperatingSystem, string Architecture, string Runtime, MemoryEvaluationRequest? Request,
    OptimizationPackage? Package, MemoryPhase[] Phases, MemoryQualityComparison[] Comparisons,
    double ProcessWallMilliseconds, ObservedProcessMetrics? ProcessMetrics, string[] Limitations);

[JsonSerializable(typeof(MemoryEvaluationRequest))]
[JsonSerializable(typeof(MemoryEvaluationEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, UseStringEnumConverter = true)]
internal sealed partial class MemoryEvaluationJsonContext : JsonSerializerContext;
