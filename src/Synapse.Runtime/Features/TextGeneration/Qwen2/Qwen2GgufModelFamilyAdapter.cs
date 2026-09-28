using ManagedCode.Synapse.Runtime.Features.ModelLoading;
using ManagedCode.Synapse.Runtime.Features.ModelLoading.Gguf;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration.Qwen2;

internal sealed class Qwen2GgufModelFamilyAdapter : IGgufModelFamilyAdapter
{
    public string Architecture => "qwen2";

    public ITextGenerationModel Load(GgufFile file, ModelLoadOptions options) =>
        Qwen2Model.Load(file, options);
}
