using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, eventArgs) =>
{
    eventArgs.Cancel = true;
    cancellation.Cancel();
};

return await FoundryCommandLine.RunAsync(args, cancellation.Token).ConfigureAwait(false);

internal static class FoundryCommandLine
{
    private const string Usage =
        "Usage: Synapse.FoundryLocalBenchmarks <command> [options]\n" +
        "  plan   --set <set.json> [--summary <path>]\n" +
        "  fetch  --set <set.json> --alias <alias> --cache <dir> [--device cpu|gpu]\n" +
        "  run    --set <set.json> --alias <alias> --cache <dir> --scenario <1-or-3-turn.json> " +
        "--output <new.json> [--device cpu|gpu] [--max-tokens 64] [--warmups 1] [--measurements 3] " +
        "[--runner-label <text>]\n" +
        "  report --input <raw.json> [--summary <path>]";

    public static async Task<int> RunAsync(string[] args, CancellationToken cancellationToken)
    {
        var arguments = args.Length == 0 ? null : FoundryArguments.Parse(args[1..]);
        if (arguments is null)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        try
        {
            return args[0] switch
            {
                "plan" => await FoundryPlanCommand.RunAsync(arguments).ConfigureAwait(false),
                "fetch" => await FoundryFetchCommand.RunAsync(arguments, cancellationToken).ConfigureAwait(false),
                "run" => await FoundryRunCommand.RunAsync(arguments, cancellationToken).ConfigureAwait(false),
                "report" => await FoundryReportCommand.RunAsync(arguments).ConfigureAwait(false),
                _ => throw new FoundryUsageException($"Unknown command '{args[0]}'."),
            };
        }
        catch (FoundryUsageException exception)
        {
            Console.Error.WriteLine(exception.Message);
            Console.Error.WriteLine(Usage);
            return 2;
        }
        catch (FoundryNotCachedException exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 4;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }
}

internal sealed class FoundryUsageException(string message) : Exception(message);

internal sealed class FoundryNotCachedException(string message) : Exception(message);
