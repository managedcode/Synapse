using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

namespace ManagedCode.Synapse.Cli.Features.ModelPackages;

internal static class CompiledModelCommand
{
    internal static async Task<int> RunAsync(IReadOnlyList<string> arguments)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 1; index < arguments.Count; index += 2)
        {
            if (index + 1 >= arguments.Count || !values.TryAdd(arguments[index], arguments[index + 1]))
            {
                return Usage();
            }
        }

        if (arguments[0] == "inspect" && values.Count == 1 && values.TryGetValue("--model", out var path))
        {
            Print(CompiledPackageReader.Inspect(path));
            return 0;
        }

        if (arguments[0] != "compile" || values.Count != 2 || !values.TryGetValue("--source", out var source) ||
            !values.TryGetValue("--output", out var destination))
        {
            return Usage();
        }

        using var cancellation = new CancellationTokenSource();
        void Cancel(object? sender, ConsoleCancelEventArgs eventArgs)
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        }

        Console.CancelKeyPress += Cancel;
        try
        {
            Console.Error.WriteLine("Compiling lossless Qwen2 tensor layout...");
            Print(await CompiledPackageCompiler.CompileAsync(source, destination, cancellation.Token).ConfigureAwait(false));
            return 0;
        }
        finally
        {
            Console.CancelKeyPress -= Cancel;
        }
    }

    private static void Print(CompiledPackageInfo info) =>
        Console.WriteLine(JsonSerializer.Serialize(info, CompiledModelJsonContext.Default.CompiledPackageInfo));

    private static int Usage()
    {
        Console.Error.WriteLine("Usage: synapse model compile --source <model.gguf> --output <model.synapse> | synapse model inspect --model <model.synapse>");
        return 2;
    }
}

[JsonSerializable(typeof(CompiledPackageInfo))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class CompiledModelJsonContext : JsonSerializerContext;
