using ManagedCode.Synapse.Runtime.Features.Tokenization;

namespace ManagedCode.Synapse.Cli.Features.Tokenization;

/// <summary>
/// <c>synapse tokenize</c> writes the token IDs of a UTF-8 text file (one per line); <c>synapse detokenize</c>
/// writes the text of a token-ID file. Both use the model's own GGUF vocabulary (ADR-014).
/// </summary>
internal static class TokenizeCommand
{
    private const string Usage =
        "Usage: synapse tokenize --model <model.gguf> --text-file <path> [--no-parse-special]\n" +
        "       synapse detokenize --model <model.gguf> --tokens-file <path> [--special]";

    public static int Run(string command, IReadOnlyList<string> arguments)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var flags = new HashSet<string>(StringComparer.Ordinal);
        for (var index = 0; index < arguments.Count; index++)
        {
            if (arguments[index] is "--no-parse-special" or "--special")
            {
                _ = flags.Add(arguments[index]);
            }
            else if (index + 1 < arguments.Count && arguments[index].StartsWith("--", StringComparison.Ordinal))
            {
                values[arguments[index]] = arguments[++index];
            }
            else
            {
                return Fail();
            }
        }

        var input = command == "tokenize" ? "--text-file" : "--tokens-file";
        if (!values.TryGetValue("--model", out var model) || !values.TryGetValue(input, out var path) || !File.Exists(path))
        {
            return Fail();
        }

        try
        {
            var tokenizer = TextTokenizers.FromGguf(model);
            if (command == "tokenize")
            {
                var ids = tokenizer.Encode(File.ReadAllText(path), parseSpecialTokens: !flags.Contains("--no-parse-special"));
                Console.Out.Write(string.Join('\n', ids) + "\n");
                return 0;
            }

            var tokens = File.ReadAllText(path)
                .Split([',', ' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
                .Select(value => int.Parse(value, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray();
            Console.Out.Write(tokenizer.Decode(tokens, includeSpecialTokens: flags.Contains("--special")));
            return 0;
        }
        catch (Exception exception) when (exception is FormatException or NotSupportedException or InvalidDataException
            or IOException or ArgumentOutOfRangeException or OverflowException)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static int Fail()
    {
        Console.Error.WriteLine(Usage);
        return 2;
    }
}
