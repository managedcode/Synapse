using ManagedCode.Synapse.Runtime.Features.ModelLoading;

namespace ManagedCode.Synapse.Runtime.Features.TextGeneration;

/// <summary>Reads the key of one cached position for one KV head.</summary>
internal delegate ReadOnlySpan<float> KeyReader(int position);

/// <summary>
/// The reference semantics of query-aware KV page activation (ADR-016). For one decode query of one KV head it
/// marks the attended positions: the sink page, the pages overlapping the recent window, and the budgeted pages.
/// Key-bound selection ranks a page by <c>max over the group's heads of Σ_d max(q_d·min_d, q_d·max_d)</c>, an upper
/// bound of every query-key product in the page.
/// </summary>
internal static class KvPageSelector
{
    /// <summary>Marks <c>selected[0..position]</c>; <paramref name="queries"/> holds the group's query heads.</summary>
    public static void Select(
        KvPageActivation activation,
        ReadOnlySpan<float> queries,
        int groupHeads,
        int headDimension,
        KeyReader keys,
        int position,
        int seed,
        Span<bool> selected)
    {
        var count = position + 1;
        var size = activation.PageTokens;
        selected[..count].Clear();
        var windowPage = Math.Max(0, count - activation.WindowTokens) / size;
        Mark(selected, 0, size, position);
        for (var page = windowPage; page <= position / size; page++)
        {
            Mark(selected, page, size, position);
        }

        var candidates = Math.Max(0, windowPage - 1);
        if (candidates <= activation.BudgetPages)
        {
            for (var page = 1; page < windowPage; page++)
            {
                Mark(selected, page, size, position);
            }

            return;
        }

        foreach (var page in activation.Selection == KvPageSelection.Random
            ? RandomPages(candidates, activation.BudgetPages, seed)
            : BestPages(queries, groupHeads, headDimension, keys, candidates, activation.BudgetPages, size))
        {
            Mark(selected, page, size, position);
        }
    }

    /// <summary>Upper bound of <c>q·k</c> over positions <c>first..last</c> and the group's query heads.</summary>
    public static float PageBound(
        ReadOnlySpan<float> queries,
        int groupHeads,
        int headDimension,
        KeyReader keys,
        int first,
        int last)
    {
        Span<float> minimum = stackalloc float[headDimension];
        Span<float> maximum = stackalloc float[headDimension];
        keys(first).CopyTo(minimum);
        keys(first).CopyTo(maximum);
        for (var position = first + 1; position <= last; position++)
        {
            var key = keys(position);
            for (var channel = 0; channel < headDimension; channel++)
            {
                minimum[channel] = MathF.Min(minimum[channel], key[channel]);
                maximum[channel] = MathF.Max(maximum[channel], key[channel]);
            }
        }

        var best = float.NegativeInfinity;
        for (var head = 0; head < groupHeads; head++)
        {
            var query = queries.Slice(head * headDimension, headDimension);
            var bound = 0f;
            for (var channel = 0; channel < headDimension; channel++)
            {
                bound += MathF.Max(query[channel] * minimum[channel], query[channel] * maximum[channel]);
            }

            best = MathF.Max(best, bound);
        }

        return best;
    }

    /// <summary>Pages <c>1..candidates</c> ranked by bound; ties keep the earlier page.</summary>
    private static int[] BestPages(
        ReadOnlySpan<float> queries,
        int groupHeads,
        int headDimension,
        KeyReader keys,
        int candidates,
        int budget,
        int size)
    {
        var ranked = new (float Bound, int Page)[candidates];
        for (var page = 1; page <= candidates; page++)
        {
            var first = page * size;
            ranked[page - 1] = (PageBound(queries, groupHeads, headDimension, keys, first, first + size - 1), page);
        }

        Array.Sort(ranked, static (left, right) =>
            left.Bound != right.Bound ? right.Bound.CompareTo(left.Bound) : left.Page.CompareTo(right.Page));
        return [.. ranked.Take(budget).Select(entry => entry.Page)];
    }

    /// <summary>A seeded partial Fisher–Yates draw of <paramref name="budget"/> pages from <c>1..candidates</c>.</summary>
    private static int[] RandomPages(int candidates, int budget, int seed)
    {
        var pages = Enumerable.Range(1, candidates).ToArray();
        var random = new Random(seed);
        for (var index = 0; index < budget; index++)
        {
            var swap = random.Next(index, pages.Length);
            (pages[index], pages[swap]) = (pages[swap], pages[index]);
        }

        return pages[..budget];
    }

    private static void Mark(Span<bool> selected, int page, int size, int position)
    {
        var first = page * size;
        var last = Math.Min(first + size - 1, position);
        selected[first..(last + 1)].Fill(true);
    }
}
