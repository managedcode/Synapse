using System.Text.Json;
using System.Text.Json.Serialization;

/// <summary>Immutable raw evidence for one Foundry Local model on one machine (ADR-011).</summary>
internal sealed record FoundryEvidence(
    string Status,
    string Subject,
    string SdkPackage,
    string SdkVersion,
    string SdkAssemblyVersion,
    string OnnxRuntimeVersion,
    string OnnxRuntimeGenAiVersion,
    string NativeRuntimeSha256,
    string ModelSetId,
    string ModelSetSha256,
    string ScenarioSha256,
    string Family,
    string Alias,
    string VariantId,
    string Device,
    string ExecutionProvider,
    int CatalogFileSizeMb,
    string License,
    string? GenaiModelType,
    JsonElement? GenaiSearch,
    IReadOnlyList<FoundryModelFile> ModelFiles,
    string RunnerLabel,
    string OperatingSystem,
    string Architecture,
    int ProcessorCount,
    long TotalMemoryBytes,
    string ThreadPolicy,
    string SamplingPolicy,
    string SystemPromptMode,
    int ContextTokens,
    int OriginalMaxLength,
    string ContextMode,
    int MaxTokens,
    int Warmups,
    int Measurements,
    double ManagerInitMilliseconds,
    double CatalogMilliseconds,
    double LoadMilliseconds,
    long? PeakResidentBytes,
    long? PeakPhysicalFootprintBytes,
    int MemorySampleCount,
    IReadOnlyList<FoundryTurn> Turns,
    IReadOnlyList<FoundrySample> Samples);

internal sealed record FoundryModelFile(string Path, long SizeBytes, string Sha256);

internal sealed record FoundryTurn(int Number, string System, string User, string LockedAssistant);

internal sealed record FoundrySample(
    int Turn,
    int Round,
    bool Warmup,
    string QualityStatus,
    int PromptTokens,
    int GeneratedTokens,
    int StreamedItems,
    string FinishReason,
    string Text,
    string ReasoningText,
    double TimeToFirstTokenMilliseconds,
    double RequestWallMilliseconds,
    double? DecodeTokensPerSecond,
    double ProcessCpuMilliseconds,
    long? ResidentBytes,
    long? PhysicalFootprintBytes);

[JsonSerializable(typeof(FoundryEvidence))]
[JsonSerializable(typeof(FoundryMatrix))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class FoundryJsonContext : JsonSerializerContext;
