using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;

internal static class SourcePreparationCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        try
        {
            var options = SourcePreparationOptions.Parse(args);
            if (options is null)
            {
                Console.Error.WriteLine("Usage: source-decode --model <model.safetensors> --tensor <name> [--samples 30]");
                return 2;
            }

            var evidence = await MeasureAsync(options).ConfigureAwait(false);
            Console.WriteLine(JsonSerializer.Serialize(evidence, SourcePreparationJsonContext.Default.SourcePreparationEvidence));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static async Task<SourcePreparationEvidence> MeasureAsync(SourcePreparationOptions options)
    {
        var tensor = SafeTensorHeaderReader.Read(options.Model).Tensors.Single(tensor => tensor.Name == options.Tensor);
        if (tensor.ElementCount > 4_194_304 || !SourceEncodings.TryGetSafeTensors(tensor.DataType.ToString(), out var decoder))
        {
            throw new NotSupportedException("The source diagnostic requires a supported float tensor of at most 4,194,304 elements.");
        }

        await using var stream = File.OpenRead(options.Model);
        var modelHash = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream).ConfigureAwait(false));
        var source = new byte[checked((int)tensor.ByteLength)];
        stream.Position = tensor.Offset;
        await stream.ReadExactlyAsync(source).ConfigureAwait(false);
        var destination = new float[checked((int)tensor.ElementCount)];
        for (var warmup = 0; warmup < 3; warmup++)
        {
            decoder!.Decode(source, destination);
        }

        var samples = new SourcePreparationSample[options.Samples];
        for (var index = 0; index < samples.Length; index++)
        {
            var allocated = GC.GetAllocatedBytesForCurrentThread();
            var start = Stopwatch.GetTimestamp();
            decoder!.Decode(source, destination);
            var elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var bytes = GC.GetAllocatedBytesForCurrentThread() - allocated;
            samples[index] = new SourcePreparationSample(index, elapsed, bytes);
        }

        return new SourcePreparationEvidence(
            "scalar_source_conversion_hot_tensor", Path.GetFullPath(options.Model), modelHash,
            tensor.Name, [.. tensor.Shape], decoder!.EncodingId,
            Convert.ToHexStringLower(SHA256.HashData(source)),
            Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(destination.AsSpan()))),
            RuntimeInformation.FrameworkDescription, RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(), 3, samples);
    }
}

internal sealed record SourcePreparationOptions(string Model, string Tensor, int Samples)
{
    public static SourcePreparationOptions? Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (index + 1 >= args.Length || args[index] is not ("--model" or "--tensor" or "--samples") ||
                !values.TryAdd(args[index], args[index + 1]))
            {
                return null;
            }
        }

        if (!values.TryGetValue("--model", out var model) || !values.TryGetValue("--tensor", out var tensor) ||
            string.IsNullOrWhiteSpace(model) || string.IsNullOrWhiteSpace(tensor) ||
            !int.TryParse(values.GetValueOrDefault("--samples", "30"), NumberStyles.None, CultureInfo.InvariantCulture, out var samples) ||
            samples is < 30 or > 1000)
        {
            return null;
        }

        return new SourcePreparationOptions(model, tensor, samples);
    }
}

internal sealed record SourcePreparationSample(int Index, double Milliseconds, long AllocatedBytes);

internal sealed record SourcePreparationEvidence(
    string Scenario, string ModelPath, string ModelSha256, string Tensor, long[] Shape, string Encoding,
    string SourceTensorSha256, string DecodedTensorSha256, string Runtime, string OperatingSystem,
    string Architecture, int Warmup, SourcePreparationSample[] Samples);

[JsonSerializable(typeof(SourcePreparationEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class SourcePreparationJsonContext : JsonSerializerContext;
