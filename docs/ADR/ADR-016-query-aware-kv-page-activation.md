# ADR-016: Query-aware KV page activation (partial activation over the context)

Status: Accepted for the CPU reference semantics; the GPU implementation is pending. Date: 2026-09-28.
Extends ADR-002, ADR-003, ADR-012, and ADR-015.

## Context

ADR-002 makes partial activation the central direction: a request activates
only the regions it needs. Qwen2.5 is a dense checkpoint, so skipping its
blocks changes its function. Any skip there is an approximation that needs
quality evidence (ADR-003 `ApproximateProvenance`).

The owner proposed applying the same idea to the context. Tokens are already
vectors, so the KV cache can be searched like an embedding index. Near,
relevant entries are read first; distant, irrelevant ones can live in another
chunk (a colder tier).

Measurements on the M2 Pro (Metal, Qwen2.5-0.5B, 32k) support the target:

- **Prefill.** GPU time is 99.7% of wall time. A 512-token step grows from
  175 ms at an empty context to 995 ms at 28k, and that growth is attention.
- **Decode.** At 32k with FP16 KV, the KV reads per token (402 MB) are about
  43% of the bytes a token reads; the weights are the rest.

Attention is a soft maximum-inner-product search: each query weights every
key by `exp(q·k)`. Published results show that most of the mass falls on a
few positions:
- attention sinks (StreamingLLM, arXiv 2309.17453);
- a recent window;
- a small, query-dependent set of distant blocks (Quest, arXiv 2406.10774;
  InfLLM, arXiv 2402.04617; RetrievalAttention, arXiv 2409.10516).

These papers are references for the design, not evidence for Synapse.

## Decision

- **The searched vectors are keys, per layer and per KV head.** They are not
  token embeddings: attention scores use the current query against that
  layer's keys. The search is by inner product, not by distance.
- **Pages are the regions.** The KV slot is cut into fixed pages of 16, 32, or
  64 positions (`PageTokens`, default 64). The CPU evidence favours 16-token
  pages at equal token budgets. Each (layer, KV head, page) keeps a summary: the per-channel
  minimum and maximum of its keys. The kernel that writes K updates the
  summary. For any query `q`, the page's score upper bound is
  `Σ_d max(q_d·min_d, q_d·max_d)`. A grouped-query KV head takes the maximum
  of the bound over its query heads.
- **Activation rule.** A decode step always attends to:
  - the sink page (positions 0..63);
  - the last `window` positions;
  - the `budget` pages with the highest bound among the rest.

  The rule is a `ProfileDecision("kv-pages:<budget>:<window>")` with
  `ApproximateProvenance`. Its skip semantics: an unselected page contributes
  nothing to that step's softmax. Nothing is deleted, and later steps may
  select it again.
- **Exactness boundary.** When `budget × 64 + window + 64` covers the whole
  prefix, the selection is every page and the result must equal dense
  attention within the existing decode tolerance. A test pins this.
- **Scope of the first step.** Decode only. Prefill stays dense, so the whole
  prompt is read once. Every batched step names its prompt boundary
  (`IBatchDecoder.Forward(tokens, promptStart)`): prompt tokens stay dense even
  when one lands alone in a step, as the last token of a 641-token prompt in
  64-token chunks or a one-token reused tail (ADR-018) does. Only tokens before
  the boundary that are alone in their slot are decode units. The C# managed and native CPU backends implement it
  first as the reference semantics (`KvPageSelector`). The reference backend
  and GPU backends reject it at load until they implement it. The Metal
  kernels (page summaries updated with K, a bound-and-select kernel, and
  decode attention over a page list) follow and must match the C# selection.
- **Opt-in.** `ModelLoadOptions.KvPageActivation` defaults to `null` (dense).
  The CLI option is `--kv-pages <budget>:<window>[:random[:seed]]`. The runtime profile gains
  a `+kvpages<budget>w<window>` suffix, so evidence can never confuse it with
  dense.
- **Qualification before claims** (ADR-015). Evidence reports three axes
  against dense on identical tokens:
  - quality: decode-mode perplexity, needle, multi-key, and variable-tracking
    accuracy;
  - memory: the KV bytes read per token, and the resident KV once cold pages
    move to a lower tier;
  - speed: decode tok/s and full generation time.

  A budget-matched random page selection is the control that ADR-002
  requires.
- **Cold tier later.** Unselected pages may move to host memory or disk, with
  page metadata in ZoneTree. That is a separate step with its own ADR update,
  after the in-GPU selection is qualified.

## Consequences

- The dense path is unchanged and stays the default and the reference.
- An expected quality risk: a needle far from the recent window must win the
  bound ranking. The needle and multi-key tasks exist to catch that.
- Summaries cost `2 × 64 × 4` bytes per page per layer per KV head (about
  12 MiB at 32k for this model). Selection costs one bound per page per step.
- Tests: `KeyBoundNeverUnderestimatesAPageScore`,
  `SelectionKeepsSinkWindowAndTheBestPage`,
  `RandomControlSelectsTheBudgetDeterministically`,
  `KvPagesCoveringThePrefixEqualDense`,
  `PromptTokensStayDenseEvenAloneInAStep`, and
  `KvPagesAreNamedAndRejectedWhereNotImplemented`. Quality comes from
  decode-mode scoring (`synapse score --scoring-rows 1 --kv-pages ...`), which
  sends every scored position through the decode path.
