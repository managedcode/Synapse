# ADR-018: Prompt prefix reuse on the direct session

Status: Accepted. Date: 2026-09-29. Extends ADR-007 (sessions) and ADR-017 (KV capacity).

## Context

The owner wants practical large contexts. In real use a long context is read
many times: a chat about one document, or an agent that re-sends its
transcript. Each request re-prefills everything.
- **Cost.** At 32k tokens that is 27 s on the M2 Pro GPU, even when only the
  last question changed.
- **Prior art.** llama.cpp (`cache_prompt`), MLX (prompt cache), and vLLM
  (prefix caching) reuse the KV of a shared prefix instead. The 2026-09-29
  research ranked a persistent prefix cache as the largest user-visible gain.

## Decision

- **In-process reuse first.** With `ModelLoadOptions.ReusePromptPrefix`, the
  direct session (slot 0) remembers the tokens whose K and V it holds: the
  last prompt plus the generated tokens fed back. A new direct `Generate`
  evaluates only the tokens after the longest common prefix, from that
  position on.
  - At least the last prompt token is always evaluated, so the step returns
    logits.
  - `TextGenerationResult.ReusedPromptTokens` reports how many prompt tokens
    were skipped.
- **Opt-in, because numerics follow the earlier request.**
  - A reused prefix carries the K and V the earlier request computed.
  - On CPU backends every token's math is independent of how the prompt was
    chunked, so reuse is bitwise equal to a cold run. This holds with KV page
    activation too, because prompt tokens stay dense even alone in a step
    (ADR-016).
  - On Metal a short new tail runs on the matrix-vector kernel instead of the
    GEMM, so results match a cold run within the decode tolerance. Greedy
    tokens can differ only at near-ties, as with llama.cpp's prompt cache.
- **Invalidation.** Any other direct use of slot 0 (scoring, prompt logits,
  incremental evaluation) clears the remembered tokens. A request first cuts
  the remembered tokens down to the prefix it shares, before it writes any
  position. A request that fails part way, such as one cancelled from its
  progress callback, then leaves only the prefix it never overwrote. Batched
  `GenerateAsync` sessions use their own slots and do not reuse.
- **Next step (separate ADR).** A persistent prefix cache: payloads on disk
  keyed by model, tokenizer, RoPE, KV type, and prefix hash, with metadata in
  ZoneTree. It needs its own ADR because it adds a durable format.

## Consequences

- A second question about a 32k document costs a prefill of the new tokens
  only.
- Memory does not change: the KV is already resident.
- Tests: `ReusedPrefixSkipsSharedTokensAndMatchesAColdRunOnCpu`,
  `ReusedPrefixMatchesAColdRunOnMetal`,
  `IdenticalPromptReevaluatesOnlyItsLastTokenAndOtherDirectWorkInvalidates`,
  `FailedRequestKeepsOnlyThePrefixItDidNotOverwrite`, and
  `PromptTokensStayDenseEvenAloneInAStep`.
- Evidence: `benchmarks/results/2026-09-29-m2-pro-qwen2.5-0.5b-q8_0-prefix-reuse-diagnostic.json`.
  A second question about a 30,000-token document reused 30,015 prompt
  tokens. It reached its first token in 0.13 s instead of 23.8 s, with the
  same answer text.
