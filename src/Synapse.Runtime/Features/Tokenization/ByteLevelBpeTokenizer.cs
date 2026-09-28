using System.Text;
using System.Text.RegularExpressions;

namespace ManagedCode.Synapse.Runtime.Features.Tokenization;

/// <summary>
/// GPT-2-style byte-level BPE with the Qwen2 pre-tokenizer, built from a GGUF vocabulary, merge list, and token
/// types (ADR-014). Control and user-defined tokens are matched whole when special parsing is on.
/// </summary>
internal sealed partial class ByteLevelBpeTokenizer : ITextTokenizer
{
    private const int ControlType = 3;
    private const int UserDefinedType = 4;
    private readonly string[] _tokens;
    private readonly int[] _types;
    private readonly Dictionary<string, int> _ids;
    private readonly Dictionary<(string Left, string Right), int> _ranks;
    private readonly (string Text, int Id)[] _specials;
    private readonly Dictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly Lock _cacheGate = new();

    public ByteLevelBpeTokenizer(string[] tokens, int[] types, string[] merges)
    {
        if (tokens.Length != types.Length)
        {
            throw new InvalidDataException("Tokenizer token and type arrays differ in length.");
        }

        _tokens = tokens;
        _types = types;
        _ids = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (var id = 0; id < tokens.Length; id++)
        {
            _ = _ids.TryAdd(tokens[id], id);
        }

        _ranks = new Dictionary<(string, string), int>(merges.Length);
        for (var rank = 0; rank < merges.Length; rank++)
        {
            var space = merges[rank].IndexOf(' ', StringComparison.Ordinal);
            if (space <= 0)
            {
                throw new InvalidDataException($"Tokenizer merge {rank} is malformed.");
            }

            _ = _ranks.TryAdd((merges[rank][..space], merges[rank][(space + 1)..]), rank);
        }

        _specials = [.. Enumerable.Range(0, tokens.Length)
            .Where(id => types[id] is ControlType or UserDefinedType && tokens[id].Length > 0)
            .Select(id => (tokens[id], id))
            .OrderByDescending(special => special.Item1.Length)];
    }

    public int VocabularySize => _tokens.Length;

    public IReadOnlyList<int> Encode(string text, bool parseSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>(text.Length / 3);
        var start = 0;
        while (start < text.Length)
        {
            var (position, special) = parseSpecialTokens ? NextSpecial(text, start) : (text.Length, -1);
            EncodeOrdinary(text.AsSpan(start, position - start), ids);
            if (special < 0)
            {
                break;
            }

            ids.Add(special);
            start = position + _tokens[special].Length;
        }

        return ids;
    }

    public string Decode(IReadOnlyList<int> tokens, bool includeSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        var bytes = new List<byte>(tokens.Count * 4);
        var text = new StringBuilder();
        foreach (var id in tokens)
        {
            if ((uint)id >= (uint)_tokens.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), id, "Token ID is outside the vocabulary.");
            }

            if (_types[id] is ControlType or UserDefinedType)
            {
                Flush(bytes, text);
                if (includeSpecialTokens || _types[id] == UserDefinedType)
                {
                    _ = text.Append(_tokens[id]);
                }

                continue;
            }

            foreach (var character in _tokens[id])
            {
                bytes.Add(ByteUnicodeMap.ToByte(character));
            }
        }

        Flush(bytes, text);
        return text.ToString();
    }

    private static void Flush(List<byte> bytes, StringBuilder text)
    {
        if (bytes.Count > 0)
        {
            _ = text.Append(Encoding.UTF8.GetString([.. bytes]));
            bytes.Clear();
        }
    }

    private (int Position, int Id) NextSpecial(string text, int start)
    {
        var best = (Position: text.Length, Id: -1);
        foreach (var (special, id) in _specials)
        {
            var found = text.IndexOf(special, start, StringComparison.Ordinal);
            if (found >= 0 && found < best.Position)
            {
                best = (found, id);
            }
        }

        return best;
    }

    private void EncodeOrdinary(ReadOnlySpan<char> text, List<int> ids)
    {
        if (text.IsEmpty)
        {
            return;
        }

        foreach (var match in PreTokenizer().EnumerateMatches(text))
        {
            var piece = ByteUnicodeMap.Encode(text.Slice(match.Index, match.Length));
            ids.AddRange(EncodePiece(piece));
        }
    }

    private int[] EncodePiece(string piece)
    {
        lock (_cacheGate)
        {
            if (_cache.TryGetValue(piece, out var cached))
            {
                return cached;
            }
        }

        var result = _ids.TryGetValue(piece, out var whole) ? [whole] : Merge(piece);
        lock (_cacheGate)
        {
            if (_cache.Count < 1 << 20)
            {
                _cache[piece] = result;
            }
        }

        return result;
    }

    /// <summary>Repeatedly merges the adjacent pair with the lowest rank, as GPT-2 BPE does.</summary>
    private int[] Merge(string piece)
    {
        var symbols = new List<string>(piece.Length);
        for (var index = 0; index < piece.Length; index++)
        {
            var width = char.IsHighSurrogate(piece[index]) && index + 1 < piece.Length ? 2 : 1;
            symbols.Add(piece.Substring(index, width));
            index += width - 1;
        }

        while (symbols.Count > 1)
        {
            var (bestRank, bestIndex) = (int.MaxValue, -1);
            for (var index = 0; index + 1 < symbols.Count; index++)
            {
                if (_ranks.TryGetValue((symbols[index], symbols[index + 1]), out var rank) && rank < bestRank)
                {
                    (bestRank, bestIndex) = (rank, index);
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        return [.. symbols.Select(symbol => _ids.TryGetValue(symbol, out var id)
            ? id
            : throw new InvalidDataException($"BPE produced a symbol missing from the vocabulary: '{symbol}'."))];
    }

    /// <summary>The Qwen2 pre-tokenizer split (Hugging Face `tokenizer.json` and llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`).</summary>
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", RegexOptions.CultureInvariant)]
    private static partial Regex PreTokenizer();
}
