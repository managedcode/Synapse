namespace ManagedCode.Synapse.Runtime.Features.ModelLoading;

/// <summary>
/// Query-aware KV page activation (ADR-016): a decode token attends to the sink page, the pages overlapping its
/// last <see cref="WindowTokens"/> positions, and <see cref="BudgetPages"/> more pages chosen by
/// <see cref="Selection"/>. An approximation profile: it changes the model's function and needs quality evidence.
/// </summary>
/// <param name="BudgetPages">Pages of <see cref="PageTokens"/> positions selected beyond the sink and the window.</param>
/// <param name="WindowTokens">Most recent positions always attended.</param>
public sealed record KvPageActivation(int BudgetPages, int WindowTokens)
{
    /// <summary>Positions per page: 16, 32, or 64. Smaller pages give tighter key bounds and more pages to rank.</summary>
    public int PageTokens { get; init; } = 64;

    /// <summary>How the budgeted pages are chosen; <see cref="KvPageSelection.Random"/> is the ADR-002 control.</summary>
    public KvPageSelection Selection { get; init; } = KvPageSelection.KeyBound;

    /// <summary>Seed of the random control; unused by key-bound selection.</summary>
    public int Seed { get; init; }

    /// <summary>Runtime-profile suffix, for example <c>kvpages16w256</c> or <c>kvpages16w256r</c>.</summary>
    public string Name => $"kvpages{BudgetPages}w{WindowTokens}" + (PageTokens == 64 ? string.Empty : $"p{PageTokens}") +
        (Selection == KvPageSelection.Random ? "r" : string.Empty);

    /// <summary>Rejects negative budgets and windows.</summary>
    public void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegative(BudgetPages);
        ArgumentOutOfRangeException.ThrowIfNegative(WindowTokens);
        if (PageTokens is not (16 or 32 or 64))
        {
            throw new ArgumentOutOfRangeException(nameof(PageTokens), PageTokens, "KV pages hold 16, 32, or 64 positions.");
        }

        if (!Enum.IsDefined(Selection))
        {
            throw new ArgumentOutOfRangeException(nameof(Selection), Selection, "Unknown KV page selection.");
        }
    }
}

/// <summary>How KV pages beyond the sink and the window are chosen.</summary>
public enum KvPageSelection
{
    /// <summary>The pages with the highest upper bound of the query-key score (per-channel key minimum and maximum).</summary>
    KeyBound,

    /// <summary>Uniformly random pages at the same budget: the control for key-bound selection.</summary>
    Random,
}
