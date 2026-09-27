using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>Format and architecture-dispatching model loader.</summary>
public static class ModelLoader
{
    /// <summary>GGUF architecture families with executable adapters.</summary>
    public static IReadOnlyList<string> SupportedGgufArchitectures =>
        GgufModelFamilyRegistry.SupportedArchitectures;

    /// <summary>Loads a supported local model without guessing its family from a filename.</summary>
    public static ITextGenerationModel Load(string modelPath, int contextSize = 512) =>
        Load(modelPath, contextSize, Environment.ProcessorCount);

    /// <summary>Loads a supported local model with an explicit CPU parallelism limit.</summary>
    public static ITextGenerationModel Load(string modelPath, int contextSize, int maximumParallelism)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumParallelism);
        if (!string.Equals(Path.GetExtension(modelPath), ".gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                $"Model format '{Path.GetExtension(modelPath)}' is unsupported; the current executable adapter accepts GGUF.");
        }

        var file = GgufFile.Open(modelPath);
        try
        {
            return GgufModelFamilyRegistry.Resolve(file).Load(file, contextSize, maximumParallelism);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
