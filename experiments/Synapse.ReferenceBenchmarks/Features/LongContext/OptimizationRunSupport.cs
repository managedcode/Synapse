using System.Globalization;
using System.Security.Cryptography;
using System.Text;

internal static class OptimizationRunSupport
{
    public static bool Arguments(string[] args, string inputFlag, out string input, out string output)
    {
        input = output = string.Empty;
        if (args.Length != 4)
        {
            return false;
        }

        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (var index = 0; index < args.Length; index += 2)
        {
            if (!values.TryAdd(args[index], args[index + 1]))
            {
                return false;
            }
        }

        return values.Count == 2 && values.TryGetValue(inputFlag, out input!) &&
            values.TryGetValue("--output", out output!) && !string.IsNullOrWhiteSpace(input) && !string.IsNullOrWhiteSpace(output);
    }

    public static string TokenSha256(IReadOnlyList<int> tokens)
    {
        var text = new StringBuilder(tokens.Count * 7);
        foreach (var token in tokens)
        {
            _ = text.Append(token.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())));
    }

    public static async Task<int> WithConsoleCancellationAsync(Func<CancellationToken, Task<int>> run)
    {
        using var stop = new CancellationTokenSource();
        void Handler(object? sender, ConsoleCancelEventArgs eventArgs)
        {
            _ = sender;
            eventArgs.Cancel = true;
            stop.Cancel();
        }
        Console.CancelKeyPress += Handler;
        try
        {
            return await run(stop.Token).ConfigureAwait(false);
        }
        finally
        {
            Console.CancelKeyPress -= Handler;
        }
    }
}
