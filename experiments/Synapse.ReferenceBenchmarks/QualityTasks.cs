using System.Globalization;
using System.Text;
using ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// RULER-style long-context tasks with exact answers (ADR-015). Each case is generated deterministically from its seed
/// and placed inside a pinned natural-text haystack cut to a token budget.
/// </summary>
internal sealed class QualityTaskFactory
{
    internal const string SystemPrompt = "You are a helpful assistant.";
    private static readonly string[] Adjectives =
    [
        "amber", "brisk", "calm", "dusty", "eager", "faint", "gentle", "hollow", "icy", "jolly", "keen", "lucid",
        "mellow", "nimble", "olive", "proud", "quiet", "rapid", "silent", "tidy", "urban", "vivid", "wild", "young",
    ];

    private static readonly string[] Nouns =
    [
        "falcon", "harbor", "lantern", "meadow", "orchid", "pebble", "quartz", "river", "saddle", "thistle", "violin",
        "walnut", "anchor", "beacon", "canyon", "dolphin", "ember", "fjord", "glacier", "heron", "island", "jasper",
    ];

    private readonly ITextTokenizer _tokenizer;
    private readonly string[] _blocks;
    private readonly int[] _blockTokens;

    public QualityTaskFactory(ITextTokenizer tokenizer, string haystack)
    {
        _tokenizer = tokenizer;
        _blocks = haystack.Split("\n\n", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        _blockTokens = [.. _blocks.Select(block => tokenizer.Encode(block + "\n\n", parseSpecialTokens: false).Count)];
    }

    public static IReadOnlyList<string> TaskNames { get; } = ["needle", "multikey", "vartrack"];

    public QualityCase Create(string task, int targetTokens, double depth, int seed)
    {
        var random = new Random(seed);
        var (preamble, insertions, question, answers, maxTokens) = task switch
        {
            "needle" => Needle(random, depth),
            "multikey" => MultiKey(random, depth),
            "vartrack" => VariableTracking(random),
            "summary" => Summary(random, depth),
            _ => throw new ArgumentOutOfRangeException(nameof(task), task, "Unknown quality task."),
        };
        // Separators are counted, and a small margin absorbs BPE merges across block boundaries.
        const int Margin = 8;
        var fixedTokens = _tokenizer.Encode(RenderPrompt(preamble + "\n\n" + question), parseSpecialTokens: true).Count +
            insertions.Sum(insertion => _tokenizer.Encode(insertion.Text + "\n\n", parseSpecialTokens: false).Count);
        var haystack = BuildHaystack(targetTokens - fixedTokens - Margin, insertions);
        var prompt = RenderPrompt(preamble + haystack + "\n\n" + question);
        var tokens = _tokenizer.Encode(prompt, parseSpecialTokens: true);
        return new QualityCase(task, targetTokens, depth, seed, preamble + haystack + "\n\n" + question, prompt,
            [.. tokens], answers, maxTokens);
    }

    internal static string RenderPrompt(string user) =>
        ChatTemplates.Qwen(SystemPrompt, [new ChatMessage("user", user)], addGenerationPrompt: true);

    private string BuildHaystack(int budget, IReadOnlyList<Insertion> insertions)
    {
        var count = 0;
        for (var used = 0; count < _blocks.Length && used + _blockTokens[count] <= budget; count++)
        {
            used += _blockTokens[count];
        }

        if (count == _blocks.Length && budget > _blockTokens.Sum())
        {
            throw new InvalidDataException($"The haystack holds fewer than {budget} tokens.");
        }

        var text = new StringBuilder();
        var pending = insertions.OrderBy(insertion => insertion.Depth).ToList();
        for (var block = 0; block <= count; block++)
        {
            while (pending.Count > 0 && (int)Math.Round(pending[0].Depth * count) == block)
            {
                _ = text.Append(pending[0].Text).Append("\n\n");
                pending.RemoveAt(0);
            }

            if (block < count)
            {
                _ = text.Append(_blocks[block]).Append("\n\n");
            }
        }

        return text.ToString().TrimEnd();
    }

    private static TaskParts Needle(Random random, double depth)
    {
        var (key, value) = (Key(random), Number(random));
        return new TaskParts(
            "Some special magic numbers are hidden within the following text. Make sure to memorize them. " +
            "I will quiz you about the numbers afterwards.\n\n",
            [new Insertion(depth, $"The special magic number for {key} is: {value}.")],
            $"What is the special magic number for {key} mentioned in the provided text? Answer with the number only.",
            [value],
            16);
    }

    /// <summary>A needle answer first, then a long free-form summary: correctness plus a full-length generation.</summary>
    private static TaskParts Summary(Random random, double depth)
    {
        var (key, value) = (Key(random), Number(random));
        return new TaskParts(
            "Read the following text carefully. A special magic number is hidden in it.\n\n",
            [new Insertion(depth, $"The special magic number for {key} is: {value}.")],
            $"First state the special magic number for {key}. Then summarize the text above in detail.",
            [value],
            128);
    }

    private static TaskParts MultiKey(Random random, double depth)
    {
        var keys = new HashSet<string>(StringComparer.Ordinal);
        while (keys.Count < 4)
        {
            _ = keys.Add(Key(random));
        }

        var pairs = keys.Select(key => (Key: key, Value: Number(random))).ToArray();
        var insertions = pairs.Select((pair, index) => new Insertion(
            (depth + (index * 0.25)) % 1.0,
            $"One of the special magic numbers for {pair.Key} is: {pair.Value}.")).ToArray();
        return new TaskParts(
            "Some special magic numbers are hidden within the following text. Make sure to memorize them. " +
            "I will quiz you about the numbers afterwards.\n\n",
            insertions,
            $"What is the special magic number for {pairs[0].Key} mentioned in the provided text? Answer with the number only.",
            [pairs[0].Value],
            16);
    }

    private static TaskParts VariableTracking(Random random)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        while (names.Count < 4)
        {
            _ = names.Add(string.Concat(Enumerable.Range(0, 5).Select(_ => (char)('A' + random.Next(26)))));
        }

        var chain = names.ToArray();
        var value = random.Next(10_000, 100_000).ToString(CultureInfo.InvariantCulture);
        var insertions = chain.Select((name, index) => new Insertion(
            0.1 + (index * 0.25),
            index == 0 ? $"VAR {name} = {value}" : $"VAR {name} = VAR {chain[index - 1]}")).ToArray();
        return new TaskParts(
            "Memorize and track the chain of variable assignments hidden in the following text.\n\n",
            insertions,
            $"Question: Find all variables that are assigned the value {value} in the text above. " +
            "Answer with the variable names only, separated by commas.",
            chain,
            40);
    }

    private static string Key(Random random) =>
        $"{Adjectives[random.Next(Adjectives.Length)]}-{Nouns[random.Next(Nouns.Length)]}";

    private static string Number(Random random) =>
        random.Next(1_000_000, 10_000_000).ToString(CultureInfo.InvariantCulture);

    private sealed record Insertion(double Depth, string Text);

    private sealed record TaskParts(
        string Preamble,
        IReadOnlyList<Insertion> Insertions,
        string Question,
        IReadOnlyList<string> Answers,
        int MaxTokens);
}

/// <summary>One generated task instance: the user message, the rendered ChatML prompt, its token IDs, and answers.</summary>
internal sealed record QualityCase(
    string Task,
    int TargetTokens,
    double Depth,
    int Seed,
    string UserMessage,
    string Prompt,
    int[] Tokens,
    IReadOnlyList<string> Answers,
    int MaxTokens)
{
    /// <summary>Fraction of expected answers the output contains, and whether it contains all of them.</summary>
    public (double Score, bool Passed) Grade(string output)
    {
        var found = Answers.Count(answer => output.Contains(answer, StringComparison.Ordinal));
        return ((double)found / Answers.Count, found == Answers.Count);
    }
}
