using System.Security.Cryptography;
using ManagedCode.Synapse.Runtime.Features.ModelPackages;

/// <summary>Checks separate preparation before benchmarking; never converts a source during a run.</summary>
internal static class PreparedBenchmarkModel
{
    public static string Require(string modelPath)
    {
        var fullPath = Path.GetFullPath(modelPath);
        if (string.Equals(Path.GetExtension(fullPath), ".synapse", StringComparison.OrdinalIgnoreCase))
        {
            return fullPath;
        }

        // Revalidate before each sample: equal lengths/timestamps cannot prove unchanged weights.
        // Callers resolve arguments before starting the measured child process.
        return VerifySource(fullPath);
    }

    private static string VerifySource(string source)
    {
        var prepared = Path.ChangeExtension(source, ".synapse");
        if (!File.Exists(prepared))
        {
            throw new FileNotFoundException(
                "Prepare the model in a separate step before the benchmark: " +
                $"synapse model compile --source \"{source}\" --output \"{prepared}\"", prepared);
        }

        var info = CompiledPackageReader.Inspect(prepared);
        using var stream = File.OpenRead(source);
        var sourceHash = Convert.ToHexStringLower(SHA256.HashData(stream));
        if (!string.Equals(info.SourceSha256, sourceHash, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Prepared benchmark model does not match the external engines' source model.");
        }

        return prepared;
    }
}
