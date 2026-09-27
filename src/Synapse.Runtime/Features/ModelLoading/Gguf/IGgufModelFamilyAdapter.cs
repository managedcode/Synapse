using ManagedCode.Synapse.Runtime.Features.TextGeneration;

namespace ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

internal interface IGgufModelFamilyAdapter
{
    string Architecture { get; }

    ITextGenerationModel Load(GgufFile file, int contextSize, int maximumParallelism);
}
