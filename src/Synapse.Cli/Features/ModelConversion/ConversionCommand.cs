using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelConversion;

namespace ManagedCode.Synapse.Cli.Features.ModelConversion;

internal static class ConversionCommand
{
    internal static async Task<int> RunAsync(IReadOnlyList<string> arguments)
    {
        if (arguments[0] == "formats" && arguments.Count == 1)
        {
            Console.WriteLine("GGUF: lossless Qwen2 preparation; ONNX: bounded FP32 stateless graph subset; " +
                "SafeTensors: F32/F16/BF16 weights plus --graph. Native graphs support Linear, Add, Multiply, Silu, Softmax, Identity " +
                "and explicitly bounded rank-two batch dimensions. Text generation requires a supported decoder family.");
            return 0;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var bounds = new Dictionary<string, ConversionBound>(StringComparer.Ordinal);
        for (var index = 1; index < arguments.Count; index += 2)
        {
            if (index + 1 >= arguments.Count)
            {
                return Usage();
            }

            if (arguments[index] == "--dimension")
            {
                if (!ParseBound(arguments[index + 1], bounds))
                {
                    return Usage();
                }
            }
            else if (!values.TryAdd(arguments[index], arguments[index + 1]))
            {
                return Usage();
            }
        }

        return await ExecuteAsync(arguments[0], values, bounds).ConfigureAwait(false);
    }

    private static async Task<int> ExecuteAsync(string operation, Dictionary<string, string> values, Dictionary<string, ConversionBound> bounds)
    {
        if (operation == "inspect" && values.Count == 1 && bounds.Count == 0 && values.TryGetValue("--model", out var model))
        {
            Print(ModelConverter.Inspect(model));
            return 0;
        }

        if (operation == "run" && values.Count == 2 && bounds.Count == 0 && values.TryGetValue("--model", out model) &&
            values.TryGetValue("--inputs", out var inputs))
        {
            return RunGraph(model, inputs);
        }

        if (operation != "convert" || values.Keys.Any(key => key is not ("--source" or "--output" or "--graph")) ||
            !values.TryGetValue("--source", out var source) || !values.TryGetValue("--output", out var output))
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
            Console.Error.WriteLine("Converting and verifying prepared model...");
            Print(await ModelConverter.ConvertAsync(source, output, new(values.GetValueOrDefault("--graph"), bounds.Count == 0 ? null : bounds),
                cancellation.Token).ConfigureAwait(false));
            return 0;
        }
        finally
        {
            Console.CancelKeyPress -= Cancel;
        }
    }

    private static int RunGraph(string model, string inputsPath)
    {
        using var cancellation = new CancellationTokenSource();
        void Cancel(object? sender, ConsoleCancelEventArgs eventArgs)
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        }

        Console.CancelKeyPress += Cancel;
        try
        {
            return EvaluateGraph(model, inputsPath, cancellation.Token);
        }
        finally
        {
            Console.CancelKeyPress -= Cancel;
        }
    }

    private static int EvaluateGraph(string model, string inputsPath, CancellationToken cancellationToken)
    {
        using var stream = File.OpenRead(inputsPath);
        if (stream.Length > 4 * 1024 * 1024)
        {
            throw new InvalidDataException("Graph inputs JSON exceeds 4 MiB.");
        }

        var inputs = JsonSerializer.Deserialize(stream, ConversionCommandJsonContext.Default.DictionaryStringGraphInputData)
            ?? throw new InvalidDataException("Graph inputs JSON is null.");
        if (inputs.Any(input => input.Value is null || input.Value.Data is null || input.Value.Shape is null))
        {
            throw new InvalidDataException("Each graph input requires shape and data arrays.");
        }

        var result = NativeGraphPackage.Execute(model, inputs.ToDictionary(pair => pair.Key, pair => pair.Value.Data),
            inputs.ToDictionary(pair => pair.Key, pair => pair.Value.Shape), cancellationToken);
        Console.WriteLine(JsonSerializer.Serialize(result.ToDictionary(), ConversionCommandJsonContext.Default.DictionaryStringSingleArray));
        return 0;
    }

    private static bool ParseBound(string value, Dictionary<string, ConversionBound> bounds)
    {
        var parts = value.Split(['=', ':']);
        return parts.Length == 3 && !string.IsNullOrWhiteSpace(parts[0]) &&
            long.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out var minimum) &&
            long.TryParse(parts[2], NumberStyles.None, CultureInfo.InvariantCulture, out var maximum) &&
            minimum > 0 && maximum >= minimum && maximum <= 1_000_000 && bounds.TryAdd(parts[0], new(minimum, maximum));
    }

    private static void Print(ConversionPackageInfo info) =>
        Console.WriteLine(JsonSerializer.Serialize(info, ConversionCommandJsonContext.Default.ConversionPackageInfo));

    private static int Usage()
    {
        Console.Error.WriteLine("Usage: synapse model convert --source <model.gguf|model.onnx|weights.safetensors> --output <model.synapse> " +
            "[--graph graph.json] [--dimension name=min:max] | model inspect --model <model.synapse> | " +
            "model run --model <model.synapse> --inputs <inputs.json> | model formats");
        return 2;
    }
}

internal sealed record GraphInputData(long[] Shape, float[] Data);

[JsonSerializable(typeof(ConversionPackageInfo))]
[JsonSerializable(typeof(Dictionary<string, GraphInputData>))]
[JsonSerializable(typeof(Dictionary<string, float[]>))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, AllowDuplicateProperties = false)]
internal sealed partial class ConversionCommandJsonContext : JsonSerializerContext;
