using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Runtime.Features.GraphExecution.Reference;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SafeTensors;
using ManagedCode.Synapse.Runtime.Features.ModelPackages.SourceFormats;
using ManagedCode.Synapse.Runtime.Features.Quantization;

internal static class WeightStudyCommand
{
    public static async Task<int> RunAsync(string[] args)
    {
        try
        {
            var options = SourcePreparationOptions.Parse(args);
            if (options is null)
            {
                Console.Error.WriteLine("Usage: weight-study --model <model.safetensors> --tensor <matrix> [--samples 30]");
                return 2;
            }

            var source = await LoadAsync(options).ConfigureAwait(false);
            var evidence = Measure(source, options.Samples);
            Console.WriteLine(JsonSerializer.Serialize(evidence, WeightStudyJsonContext.Default.WeightStudyEvidence));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static async Task<WeightStudyTensor> LoadAsync(SourcePreparationOptions options)
    {
        var tensor = SafeTensorHeaderReader.Read(options.Model).Tensors.Single(tensor => tensor.Name == options.Tensor);
        if (tensor.Shape.Count != 2 || tensor.ElementCount > 4_194_304 ||
            !SourceEncodings.TryGetSafeTensors(tensor.DataType.ToString(), out var decoder))
        {
            throw new NotSupportedException("Weight study requires a supported float matrix of at most 4,194,304 elements.");
        }

        await using var stream = File.OpenRead(options.Model);
        var modelHash = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream).ConfigureAwait(false));
        var bytes = new byte[checked((int)tensor.ByteLength)];
        stream.Position = tensor.Offset;
        await stream.ReadExactlyAsync(bytes).ConfigureAwait(false);
        var weights = new float[checked((int)tensor.ElementCount)];
        decoder!.Decode(bytes, weights);
        return new WeightStudyTensor(Path.GetFullPath(options.Model), modelHash, tensor.Name, decoder.EncodingId,
            Convert.ToHexStringLower(SHA256.HashData(bytes)), bytes.Length,
            checked((int)tensor.Shape[0]), checked((int)tensor.Shape[1]), weights);
    }

    private static WeightStudyEvidence Measure(WeightStudyTensor source, int sampleCount)
    {
        var (rows, columns) = (source.Rows, source.Columns);
        var mean = BlockMeanCodec.Create();
        var meanBytes = new byte[checked((int)mean.GetEncodedLength(rows, columns))];
        var q4 = WeightCodecs.Q4;
        var q4Bytes = new byte[checked((int)q4.GetEncodedLength(rows, columns))];
        mean.Encode(source.Weights, rows, columns, meanBytes);
        q4.Encode(source.Weights, rows, columns, q4Bytes);
        var random = new Random(1701);
        float[] probes = [.. Enumerable.Range(0, columns * 16).Select(_ => (random.NextSingle() * 2) - 1)];
        var input = probes.AsSpan(0, columns).ToArray();
        var output = new float[rows];
        var sums = new double[mean.GetGroupsPerRow(columns)];
        var scratch = new float[rows];
        Action[] operations =
        [
            () => ReferenceLinearOperators.Multiply(input, source.Weights, [], output),
            () => q4.Multiply(q4Bytes, rows, columns, input, output),
            () => mean.Multiply(meanBytes, rows, columns, input, output, sums, scratch),
        ];
        for (var warmup = 0; warmup < 3; warmup++)
        {
            foreach (var operation in operations)
            {
                operation();
            }
        }

        var samples = MeasureRotated(operations, sampleCount);
        return new WeightStudyEvidence(
            "reference_codec_linear_hot_tensor_not_production_generation", source.Path, source.ModelHash,
            source.Name, source.Encoding, source.Hash, [rows, columns], source.Bytes, (long)source.Weights.Length * sizeof(float),
            mean.EncodingId, meanBytes.Length, q4.EncodingId, q4Bytes.Length,
            WeightSensitivity.MeasureCodec(mean, source.Weights, rows, columns, probes),
            WeightSensitivity.MeasureCodec(q4, source.Weights, rows, columns, probes),
            "not_evaluated_synthetic_activations", false, 1701, 16,
            Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(probes.AsSpan()))),
            RuntimeInformation.FrameworkDescription, RuntimeInformation.OSDescription,
            RuntimeInformation.ProcessArchitecture.ToString(), 3, samples);
    }

    private static WeightStudySample[] MeasureRotated(Action[] operations, int count)
    {
        var samples = new WeightStudySample[count];
        for (var sample = 0; sample < count; sample++)
        {
            var milliseconds = new double[operations.Length];
            for (var index = 0; index < operations.Length; index++)
            {
                var subject = (sample + index) % operations.Length;
                var start = Stopwatch.GetTimestamp();
                operations[subject]();
                milliseconds[subject] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }

            samples[sample] = new WeightStudySample(sample, milliseconds[0], milliseconds[1], milliseconds[2]);
        }

        return samples;
    }
}

internal sealed record WeightStudyTensor(string Path, string ModelHash, string Name, string Encoding,
    string Hash, long Bytes, int Rows, int Columns, float[] Weights);

internal sealed record WeightStudySample(int Index, double Fp32ScalarOracleMilliseconds,
    double Q4ScalarCodecMilliseconds, double BlockMeanCodecMilliseconds);

internal sealed record WeightStudyEvidence(
    string Scenario, string ModelPath, string ModelSha256, string Tensor, string SourceEncoding,
    string SourceTensorSha256, int[] Shape, long SourceBytes, long Fp32ReferenceBytes,
    string MeanEncoding, long MeanBytes, string Q4Encoding, long Q4Bytes,
    OutputDistortion MeanDistortion, OutputDistortion Q4Distortion,
    string QualityStatus, bool RuntimeEligible, int ProbeSeed, int ProbeCount, string ProbeSha256,
    string Runtime, string OperatingSystem, string Architecture, int Warmup, WeightStudySample[] Samples);

[JsonSerializable(typeof(WeightStudyEvidence))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal sealed partial class WeightStudyJsonContext : JsonSerializerContext;
