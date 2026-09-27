using ManagedCode.Synapse.Contracts.Features.Bootstrap;
using ManagedCode.Synapse.Runtime.Features.Bootstrap;

namespace ManagedCode.Synapse.IntegrationTests.Features.Bootstrap;

public sealed class RepositoryDoctorTests
{
    private const long OneGibibyte = 1L << 30;

    [Test]
    public async Task DoctorDetectsCpu()
    {
        var result = RepositoryDoctor.Inspect(new DoctorRequest(OneGibibyte));

        await Assert.That(result.IsSupported).IsTrue();
        await Assert.That(result.Architecture).IsNotEmpty();
        await Assert.That(result.ProcessArchitecture).IsNotEmpty();
        await Assert.That(result.MemoryBudgetBytes).IsEqualTo(OneGibibyte);
    }

    [Test]
    [Arguments(0L)]
    [Arguments(-1L)]
    public async Task InvalidBudgetFails(long invalidBudget)
    {
        var error = RepositoryDoctor.Inspect(new DoctorRequest(invalidBudget)).Error;

        await Assert.That(error).IsNotNull();
        await Assert.That(error!.Code).IsEqualTo(DoctorErrorCode.InvalidMemoryBudget);
    }

    [Test]
    public async Task ZoneTreeDependencyPersistsAcrossReopen()
    {
        var directory = Path.Combine(
            Path.GetTempPath(),
            $"synapse-zonetree-{Guid.NewGuid():N}");

        try
        {
            var result = ZoneTreeDependencyProbe.VerifyDurableRoundTrip(directory);

            await Assert.That(result).IsTrue();
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }

    [Test]
    public async Task ConfiguredRunnerMatchesActualPlatform()
    {
        var requiredRuntimeIdentifier = Environment.GetEnvironmentVariable(
            "SYNAPSE_REQUIRED_RUNTIME_IDENTIFIER");
        var result = RepositoryDoctor.Inspect(new DoctorRequest(OneGibibyte));

        if (!string.IsNullOrWhiteSpace(requiredRuntimeIdentifier))
        {
            await Assert.That(result.RuntimeIdentifier)
                .IsEqualTo(requiredRuntimeIdentifier);
        }

        await Assert.That(result.IsSupported).IsTrue();
    }
}
