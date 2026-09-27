using ManagedCode.Synapse.Contracts.Features.GraphExecution;

namespace ManagedCode.Synapse.Runtime.Features.GraphExecution.Validation;

/// <summary>Validates portable model IR before weights or execution memory are allocated.</summary>
public static class ModelGraphVerifier
{
    private const ushort SupportedGraphMajor = 1;
    private const ushort SupportedOpSetMajor = 1;

    /// <summary>Runs structural, shape, entry-point, and state-effect validation.</summary>
    public static GraphVerificationResult Verify(ModelGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var context = new GraphVerificationContext(graph);
        ValidateVersions(context);
        GraphDeclarationVerifier.Index(context);
        GraphStructureVerifier.Verify(context);
        GraphOperationAttributeVerifier.Verify(context);
        GraphShapeVerifier.Verify(context);
        GraphWeightVerifier.Verify(context);
        GraphStateVerifier.Verify(context);
        GraphEntryPointVerifier.Verify(context);
        GraphRegionVerifier.Verify(context);
        return new GraphVerificationResult(context.Diagnostics);
    }

    private static void ValidateVersions(GraphVerificationContext context)
    {
        if (context.Graph.GraphVersion.Major != SupportedGraphMajor)
        {
            context.Add(
                GraphDiagnosticCode.UnsupportedVersion,
                $"Graph major version {context.Graph.GraphVersion.Major} is unsupported.");
        }

        if (context.Graph.OpSetVersion.Major != SupportedOpSetMajor)
        {
            context.Add(
                GraphDiagnosticCode.UnsupportedVersion,
                $"OpSet major version {context.Graph.OpSetVersion.Major} is unsupported.");
        }
    }
}
