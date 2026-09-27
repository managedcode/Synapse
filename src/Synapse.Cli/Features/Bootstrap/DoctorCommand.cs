using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using ManagedCode.Synapse.Contracts.Features.Bootstrap;
using ManagedCode.Synapse.Runtime.Features.Bootstrap;

namespace ManagedCode.Synapse.Cli.Features.Bootstrap;

internal static class DoctorCommand
{
    private const int SuccessExitCode = 0;
    private const int InvalidConfigurationExitCode = 2;
    private const long DefaultMemoryBudgetBytes = 1L << 30;

    public static int Run(IReadOnlyList<string> arguments)
    {
        var parseResult = DoctorOptions.Parse(arguments);
        if (parseResult.Error is not null)
        {
            Console.Error.WriteLine(parseResult.Error);
            return InvalidConfigurationExitCode;
        }

        var options = parseResult.Options!;
        var result = RepositoryDoctor.Inspect(new DoctorRequest(options.MemoryBudgetBytes));
        var zoneTreeVerified = options.StateDirectory is null
            || ZoneTreeDependencyProbe.VerifyDurableRoundTrip(options.StateDirectory);

        var output = new DoctorOutput(result, zoneTreeVerified);
        Console.WriteLine(JsonSerializer.Serialize(output, DoctorJsonContext.Default.DoctorOutput));
        return result.IsSupported && zoneTreeVerified
            ? SuccessExitCode
            : InvalidConfigurationExitCode;
    }

    private sealed record DoctorOptions(long MemoryBudgetBytes, string? StateDirectory)
    {
        public static DoctorOptionsParseResult Parse(IReadOnlyList<string> arguments)
        {
            var memoryBudget = DefaultMemoryBudgetBytes;
            string? stateDirectory = null;

            for (var index = 0; index < arguments.Count; index++)
            {
                switch (arguments[index])
                {
                    case "--memory-budget-bytes" when index + 1 < arguments.Count:
                        if (!long.TryParse(
                                arguments[++index],
                                NumberStyles.None,
                                CultureInfo.InvariantCulture,
                                out memoryBudget))
                        {
                            return new(null, "--memory-budget-bytes must be a signed 64-bit integer.");
                        }

                        break;
                    case "--state-directory" when index + 1 < arguments.Count:
                        stateDirectory = arguments[++index];
                        break;
                    default:
                        return new(null, $"Unknown or incomplete doctor option '{arguments[index]}'.");
                }
            }

            return new(new DoctorOptions(memoryBudget, stateDirectory), null);
        }
    }

    private sealed record DoctorOptionsParseResult(DoctorOptions? Options, string? Error);
}

internal sealed record DoctorOutput(DoctorResult Runtime, bool ZoneTreeVerified);

[JsonSerializable(typeof(DoctorOutput))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal sealed partial class DoctorJsonContext : JsonSerializerContext;
