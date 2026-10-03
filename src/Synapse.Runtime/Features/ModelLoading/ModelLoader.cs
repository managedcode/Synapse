using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;
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
        Load(modelPath, new ModelLoadOptions { ContextSize = contextSize });

    /// <summary>Loads a supported local model with an explicit CPU parallelism limit.</summary>
    public static ITextGenerationModel Load(string modelPath, int contextSize, int maximumParallelism) =>
        Load(modelPath, new ModelLoadOptions { ContextSize = contextSize, MaximumParallelism = maximumParallelism });

    /// <summary>Loads a supported local model with explicit limits and CPU kernel backend.</summary>
    public static ITextGenerationModel Load(string modelPath, ModelLoadOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        var extension = Path.GetExtension(modelPath);
        if (!string.Equals(extension, ".synapse", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(
                "Runtime requires a prepared .synapse model. Convert it first: " +
                "synapse model convert --source <model.gguf|model.onnx|weights.safetensors> --output <model.synapse>.");
        }

        if (NativeGraphPackage.Version(modelPath) == 2)
        {
            throw new NotSupportedException("This native graph has no qualified text-generation adapter. " +
                "Execute its graph explicitly: synapse model run --model <model.synapse> --inputs <inputs.json>.");
        }

        return LoadMapped(modelPath, options);
    }

    internal static ITextGenerationModel LoadSourceForValidation(string modelPath, int contextSize = 512) =>
        LoadSourceForValidation(modelPath, new ModelLoadOptions { ContextSize = contextSize });

    internal static ITextGenerationModel LoadSourceForValidation(string modelPath, int contextSize, int maximumParallelism) =>
        LoadSourceForValidation(modelPath, new ModelLoadOptions { ContextSize = contextSize, MaximumParallelism = maximumParallelism });

    internal static ITextGenerationModel LoadSourceForValidation(string modelPath, ModelLoadOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (!string.Equals(Path.GetExtension(modelPath), ".gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException("Source validation requires a GGUF input; product runtime requires .synapse.");
        }

        return LoadMapped(modelPath, options);
    }

    private static ITextGenerationModel LoadMapped(string modelPath, ModelLoadOptions options)
    {
        var file = GgufFile.Open(modelPath);
        try
        {
            return GgufModelFamilyRegistry.Resolve(file).Load(file, options);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
