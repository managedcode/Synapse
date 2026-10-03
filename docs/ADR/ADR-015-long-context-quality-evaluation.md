# ADR-015: Long-context quality evaluation and cross-engine parity

Status: Accepted. Date: 2026-09-28.

## Context

ADR-013 made contexts up to 131,072 tokens reachable, and the pass-key
diagnostic showed that retrieval can succeed or fail. The owner then asked
whether the model stays correct on large prompts ("is it dumb, does it answer
right") and asked for comparisons across systems. Speed evidence cannot
answer that. One pass key per length cannot separate three causes:

1. an engine bug, where Synapse computes different math than a reference
   engine on the same tokens;
2. a model limit, where every correct engine fails the same way;
3. a precision or scaling-profile effect, such as FP16 KV or YaRN.

## Decision

- **Teacher-forced scoring is a model capability.** `ITextGenerationModel.Score`
  evaluates a token sequence from position 0 in a fresh direct session. For
  every position `p` in `[firstScoredPosition, n-2]` it returns the negative
  log-likelihood of `tokens[p+1]` and the greedy (arg-max) token. Log-softmax
  uses an FP32 max subtraction with an FP64 sum of `expf`, the same expression
  as llama.cpp `llama-perplexity`.
- **Scoring rows are an explicit load option.**
  `ModelLoadOptions.ScoringRowsPerStep` (default 8, range 1..512) bounds how
  many full-vocabulary logit rows one step returns. Each row costs
  `vocabulary × 4` bytes in the host buffer and in the device buffer. The logit
  capacity is `max(MaximumConcurrentSessions, ScoringRowsPerStep)`. Batch
  decoders score in steps of that many positions on slot 0. The reference
  backend scores by prefill plus one decode per position. Logit rows keep the
  batch-invariant matrix-vector path (ADR-007).
- **Perplexity follows the llama-perplexity protocol.** The token stream is
  cut into `floor(n / n_ctx)` chunks, optionally capped by `--chunks`. Each
  chunk starts from an empty cache and scores positions
  `n_ctx/2 .. n_ctx-2`. The CLI prints the cumulative PPL after each chunk and
  the final `PPL ± error`, with error `= PPL × sqrt((E[nll²] − E[nll]²)/(count−1))`.
  No BOS is inserted, because the pinned model declares none.
- **Cross-engine parity uses identical tokens.** A comparison is valid only
  when both engines tokenize the same text to identical IDs, verified with
  `llama-tokenize --no-escape`, or both receive identical IDs directly.
  llama.cpp tools run with `--no-escape`. PPL parity compares the final and
  per-chunk values at the same context, scaling profile, and KV precision.
- **Task quality uses exact-answer synthetic tasks.** The quality command
  generates RULER-style tasks deterministically from a seed:
  - single needle;
  - multi-key needle with distractors;
  - variable tracking across hops.

  Each task has a known answer and is placed at controlled depths inside a
  natural-text haystack from the pinned repository documentation corpus.
  Every subject gets the same prompt text. Answers are graded by exact
  containment. Subjects run as separate processes.
- **Evidence class.** These runs are diagnostics (ADR-005). They record raw
  JSON, commands, token hashes, identity checks, and limitations. A claim
  that Synapse "matches" another engine requires parity on identical tokens,
  and a claim about model capability requires agreement between engines.

## Consequences

- Engine bugs show up as PPL divergence or answer disagreement on identical
  tokens. Model limits show up as the same failures in every engine.
  Precision effects show up as FP32-versus-FP16 KV differences inside Synapse.
- Scoring a 32k chunk costs one prefill of the unscored half, plus
  `n_ctx/2 / ScoringRowsPerStep` steps that each return logit rows.
- MLX runs use separately converted weights. They form a separate-weights
  cohort for task quality and never count as engine parity.
- Tests: `ScoreMatchesIncrementalLogits`, `ScoreIsBackendConsistent`,
  `ScoringRowsOptionIsValidated`, `CliScoreReportsPerplexity`, and the quality
  task generator determinism tests.

## 2026-10-04: canonical corpus line endings

Normalize documentation haystacks to LF before paragraph splitting and
tokenization. CRLF checkout text previously collapsed paragraphs and could
produce a much shorter request than the specified budget on Windows. LF
inputs retain their existing bytes and token IDs. Equivalent LF/CRLF corpora
must now generate identical prompts, token IDs and answer placements for each
seed/task. This is a benchmark-input repair, not a new model-quality result.
`CorpusLineEndingsDoNotChangeQualityCases` exercises all three task families
using the real tokenizer (`TEST-CTX-005-1`, `TASK-CTX-005`).
