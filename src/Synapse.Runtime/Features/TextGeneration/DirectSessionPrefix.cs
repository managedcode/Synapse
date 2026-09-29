namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>
/// The tokens whose K and V the direct slot holds, for prompt prefix reuse (ADR-018): the last prompt plus every
/// generated token that was fed back. Disabled instances always report nothing shared.
/// </summary>
internal sealed class DirectSessionPrefix(bool enabled)
{
    private readonly List<int> _tokens = [];

    /// <summary>
    /// Claims the tokens shared with the slot for a new request, leaving at least the last prompt token to evaluate.
    /// Everything after the shared prefix is forgotten first: the request overwrites those positions, so a request
    /// that fails part way leaves only the prefix it never touched.
    /// </summary>
    public int Claim(IReadOnlyList<int> promptTokens)
    {
        var limit = Math.Min(_tokens.Count, promptTokens.Count - 1);
        var shared = 0;
        while (shared < limit && _tokens[shared] == promptTokens[shared])
        {
            shared++;
        }

        _tokens.RemoveRange(shared, _tokens.Count - shared);
        return shared;
    }

    /// <summary>Records the prompt and the generated tokens fed back; the last generated token never is.</summary>
    public void Remember(IReadOnlyList<int> promptTokens, IReadOnlyList<int> generated)
    {
        _tokens.Clear();
        if (enabled)
        {
            _tokens.AddRange(promptTokens);
            _tokens.AddRange(generated.Take(generated.Count - 1));
        }
    }

    /// <summary>Forgets the slot contents after any other direct use of it.</summary>
    public void Clear() => _tokens.Clear();
}
