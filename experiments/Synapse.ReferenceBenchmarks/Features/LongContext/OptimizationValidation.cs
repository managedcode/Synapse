using System.Security.Cryptography;
using System.Text.Json;
using ManagedCode.Synapse.Contracts.Features.GraphExecution;
using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

internal static class OptimizationValidation
{
    public static async Task<OptimizationPlan> ReadPlanAsync(string path, CancellationToken cancellationToken)
    {
        if (new FileInfo(path).Length > 4 * 1024 * 1024)
        {
            throw new InvalidDataException("Evaluation plan exceeds 4 MiB.");
        }

        var plan = JsonSerializer.Deserialize(await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false),
            OptimizationJsonContext.Default.OptimizationPlan) ?? throw new InvalidDataException("Empty evaluation plan.");
        Validate(plan);
        var directory = Path.GetDirectoryName(Path.GetFullPath(path))!;
        return plan with
        {
            ModelPath = PreparedPath(plan.ModelPath, directory),
            HaystackPath = Path.GetFullPath(plan.HaystackPath, directory),
            Profiles = [.. plan.Profiles.Select(profile => profile with
            {
                ModelPath = profile.ModelPath is null ? null : PreparedPath(profile.ModelPath, directory),
            })],
        };
    }

    public static void Validate(OptimizationPlan plan)
    {
        if (plan.SchemaVersion != 1 || string.IsNullOrWhiteSpace(plan.ModelPath) || string.IsNullOrWhiteSpace(plan.HaystackPath) ||
            plan.Contexts is not { Length: > 0 and <= 16 } || plan.Profiles is not { Length: >= 2 and <= 32 } ||
            plan.Depths is not { Length: > 0 and <= 16 } || plan.Tasks is not { Length: > 0 and <= 3 })
        {
            throw new InvalidDataException("Evaluation requires version 1, model/haystack paths, contexts and 2..32 profiles.");
        }

        if (plan.MaximumNewTokens is < 1 or > 4096 || plan.ScoredTailTokens is < 1 or > 4096 || plan.Threads is < 1 or > 256 ||
            plan.Warmups is < 0 or > 30 || plan.Measurements is < 1 or > 100 || plan.TimeoutSeconds is < 1 or > 86400 ||
            plan.Contexts.Any(context => context is null || context.ContextSize is < 2 or > 1_048_576 ||
                context.PromptTokens < 2 || (long)context.PromptTokens + plan.MaximumNewTokens > context.ContextSize) ||
            plan.Depths.Any(depth => !double.IsFinite(depth) || depth is < 0 or > 1) ||
            plan.Tasks.Any(task => !QualityTaskFactory.TaskNames.Contains(task, StringComparer.Ordinal)) ||
            plan.Tasks.Distinct(StringComparer.Ordinal).Count() != plan.Tasks.Length)
        {
            throw new InvalidDataException("Evaluation bounds, tasks, depths or prompt/output context limits are invalid.");
        }

        foreach (var profile in plan.Profiles)
        {
            ValidateProfile(profile);
        }

        if (plan.Profiles.Select(profile => profile.Id).Distinct(StringComparer.Ordinal).Count() != plan.Profiles.Length)
        {
            throw new InvalidDataException("Evaluation profile IDs must be unique.");
        }

        var baseline = plan.Profiles.SingleOrDefault(profile => profile.Id == plan.BaselineProfile);
        if (baseline is null || baseline.ReusePromptPrefix || baseline.KvPrecision != "f32" || baseline.KvPages is not null ||
            baseline.DropLayers is not null)
        {
            throw new InvalidDataException("A unique named baseline must use original weights, FP32 KV, full layers, dense attention and reuse off.");
        }

        _ = KernelBackendNames.TryParse(baseline.Backend, out var baselineBackend);
        if (plan.Profiles.Any(profile => !KernelBackendNames.TryParse(profile.Backend, out var backend) || backend != baselineBackend))
        {
            throw new InvalidDataException("Every paired profile must use the same backend as the named baseline.");
        }

        ValidateRope(plan.RopeScaling);
    }

    public static void ValidateProfile(OptimizationProfile profile)
    {
        if (profile is null || string.IsNullOrWhiteSpace(profile.Id) || profile.Id.Length > 80 ||
            !KernelBackendNames.TryParse(profile.Backend, out var backend) || profile.KvPrecision is not ("f32" or "f16"))
        {
            throw new InvalidDataException("Profile needs a bounded ID and known backend/KV precision.");
        }

        if ((profile.KvPrecision == "f16" && !KernelBackendNames.IsGpu(backend)) ||
            (profile.KvPages is not null && backend is not (KernelBackend.Managed or KernelBackend.Native)))
        {
            throw new NotSupportedException("Profile requests FP16 KV on CPU or KV pages on a backend that does not implement them.");
        }

        profile.KvPages?.ToRuntime().Validate();
        if ((profile.DropLayers is { } layers && (layers.Length == 0 || layers.Any(layer => layer < 0) || layers.Distinct().Count() != layers.Length)) ||
            (profile.EvidenceSha256 is { } evidence && (profile.DropLayers is null || !IsDigest(evidence))))
        {
            throw new InvalidDataException("Dropped layers must be distinct nonnegative indices with an optional valid evidence digest.");
        }
    }

    public static async Task<OptimizationPackage> DescribePackageAsync(string path, CancellationToken cancellationToken)
    {
        _ = PreparedPath(path, Environment.CurrentDirectory);
        var info = CompiledPackageReader.Inspect(path, cancellationToken);
        return new OptimizationPackage(Path.GetFullPath(path), await FileSha256Async(path, cancellationToken).ConfigureAwait(false),
            info.Identity, info.SourceSha256, info.GraphFingerprint);
    }

    public static async Task<string> FileSha256Async(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false));
    }

    public static string PreparedPath(string path, string directory)
    {
        if (!path.EndsWith(".synapse", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Evaluation requires an explicitly prepared .synapse model; convert separately before execution.");
        }

        return Path.GetFullPath(path, directory);
    }

    public static void ValidateRope(RopeScaling? scaling)
    {
        if (scaling is not null)
        {
            if (scaling.Kind != RopeScalingKind.Yarn)
            {
                throw new NotSupportedException("Only an explicit YaRN profile is supported.");
            }

            _ = RopeScaling.Yarn(scaling.Factor, scaling.OriginalContextLength);
        }
    }

    private static bool IsDigest(string value) => value.Length == 64 && value.All(character => character is (>= '0' and <= '9') or (>= 'a' and <= 'f'));
}
