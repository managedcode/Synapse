internal static class FoundryFetchCommand
{
    public static async Task<int> RunAsync(FoundryArguments arguments, CancellationToken cancellationToken)
    {
        arguments.AllowOnly("--set", "--alias", "--cache", "--device");
        var set = FoundryModelSet.Load(arguments.Require("--set"));
        var pinned = set.GetModel(arguments.Require("--alias")).Variant(arguments.Device());
        var manager = await FoundryRuntime.StartAsync(arguments.Require("--cache"), cancellationToken)
            .ConfigureAwait(false);
        if (pinned.Device == "gpu")
        {
            await FoundryRuntime.RequireGpuProviderAsync(manager, arguments.Require("--cache"),
                allowDownload: true, cancellationToken).ConfigureAwait(false);
        }

        var variant = await FoundryRuntime.ResolveAsync(manager, pinned, cancellationToken).ConfigureAwait(false);
        if (await variant.IsCachedAsync(cancellationToken).ConfigureAwait(false))
        {
            Console.Error.WriteLine($"{pinned.Id} is already cached.");
        }
        else
        {
            var reported = -10f;
            Console.Error.WriteLine($"Downloading {pinned.Id} ({pinned.FileSizeMb} MB)...");
            await variant.DownloadAsync(percent =>
            {
                if (percent - reported >= 10 || percent >= 100)
                {
                    reported = percent;
                    Console.Error.WriteLine($"{pinned.Id}: {percent:F0}%");
                }
            }, cancellationToken).ConfigureAwait(false);
        }

        if (!await variant.IsCachedAsync(cancellationToken).ConfigureAwait(false))
        {
            throw new InvalidOperationException($"{pinned.Id} is still not cached after download.");
        }

        var modelPath = await variant.GetPathAsync(cancellationToken).ConfigureAwait(false);
        var originalMaxLength = await FoundryContextBound.ApplyAsync(modelPath, set.ContextTokens, cancellationToken)
            .ConfigureAwait(false);
        Console.Error.WriteLine($"{pinned.Id}: search.max_length {originalMaxLength} -> " +
            $"{Math.Min(set.ContextTokens, originalMaxLength)} (original kept as {FoundryContextBound.OriginalName}).");
        Console.WriteLine(modelPath);
        return 0;
    }
}
