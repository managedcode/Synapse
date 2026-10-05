# ADR-020: Exact speculative decoding with an external draft

Status: Accepted. Date: 2026-09-29. Implements FlyBrain F4.1–F4.2 (SPC-001) and
follows the speculative-branch rules of ADR-003.

## Context

The FlyBrain direction is to run less of the big model per committed token
without changing its output.

- **Cost of decode.** Decode is bandwidth-bound: Qwen2.5-7B-Instruct-1M Q8_0
  reads 8 GB of weights per token and reaches about 17.5 tokens/s on the M2
  Pro, the same as llama.cpp. Evaluating eight tokens in one pass costs about
  the same weight traffic as evaluating one.
- **A matching draft already exists.** Qwen2.5-0.5B shares the 7B's
  tokenizer: 400 KB of text encodes to the same 95,851 IDs with both. It
  decodes at about 150 tokens/s.
- **Why greedy speculation is exact.** Greedy speculative decoding (Leviathan
  et al., arXiv 2211.17192) commits only tokens the target itself would pick,
  so its output equals the target's own greedy output.

## Decision

- **API.** `SpeculativeDecoding.Generate(target, draft, prompt,
  maximumNewTokens, draftTokens)` returns the target's greedy continuation
  plus counts: target passes, drafted tokens, and accepted tokens.
- **Loop.**
  1. The target prefills the prompt and commits its first token.
  2. Each round, the draft proposes `k` tokens greedily from its own KV.
  3. The target evaluates the last committed token plus the `k` drafts in one
     batched step on its direct slot, with a logits row per position.
  4. The longest prefix of drafts that equals the target's arg-max is
     accepted. The target's next arg-max is committed as well, so every
     target pass commits at least one token.
- **State (ADR-003 branch semantics without a new store).**
  - KV slots are indexed by position, and attention reads only positions up
    to the current one.
  - Rejected drafts leave stale K/V only beyond the committed length. The
    next round overwrites them before any read. This is the
    `RollbackTail` of the working branch.
  - The committed prefix is always complete.
  - The draft catches up on committed tokens it has not yet seen before it
    drafts again.
  - Both models' prompt-prefix memories (ADR-018) are cleared, because a
    speculative run leaves them stale.
- **Identity checks.** These fail before any work:
  - The draft's token IDs must be a prefix of the target's: equal
    `tokenizer.ggml.tokens`, types, and merges over the draft vocabulary,
    compared by SHA-256. Mismatches fail with `NotSupportedException`.
  - The draft's vocabulary must not be larger than the target's.
  - The target must support batched steps, so the reference backend fails
    explicitly.
  - `k + 1` must fit the target's logits rows.
  - If the target commits an ID the draft cannot embed, the rest runs
    target-only, which is still exact.
- **Exactness by backend.**
  - On CPU backends a token's math does not depend on how many tokens share
    its step, so speculative output is bitwise the target's greedy output.
  - On Metal, verification runs `k + 1` rows through the multi-token
    matrix-vector kernel. The result equals single-token decode within the
    decode tolerance, and tokens can differ only at near-ties, as with
    prefix reuse (ADR-018).

## 2026-10-05 adaptive depth amendment (`REQ-SPC-005`)

Add an opt-in `adaptiveDepth` argument (default false) and CLI
`--draft-tokens auto` (maximum three proposals). CPU callers may explicitly
choose a cap through seven; adaptive GPU caps above three fail before generation.
Fixed depths remain unchanged.
This applies the measured-cost idea described in TensorFold commit
`bcb8f01` independently to Synapse's external draft; it does not import MTP
heads, upstream code or upstream performance claims.

The request-local controller samples depths zero through the configured cap
twice, then compares expected committed tokens per millisecond using an EWMA
of actual committed counts and draft-plus-verification wall time at each depth.
Draft timing includes catch-up after rejection or plain rounds. Depth zero
uses the target's ordinary single-token Decode, leaves the draft idle and is
the baseline. A draft must beat that baseline by 5%; ties prefer less drafting.
Reprobes rotate across depths after 16, 32, 64 and then 128 exploitation rounds.
Clipped end-of-request rounds are excluded from learning. Costs are scoped to
this request/model/backend/context and never persisted or shared across devices.
Counters include exploration, and per-depth counts and timing totals are exposed.

All commits still come from the target. Online exploration and draft prefill
can lose time on short requests; this is an experimental opt-in until real
paired model/hardware evidence qualifies it. Startup calibration, trained MTP
heads and a confidence-based per-level continuation policy remain future work.

The first real Qwen2.5-0.5B Metal pilot with an adaptive cap of seven diverged
from dense at output position 36 in all three measured rounds. Five or more
verification rows enter the GEMM projection path and lack the matrix-vector
path's batch invariance. Reject those adaptive GPU windows explicitly; the
CLI samples only depths zero through three. Do not count the rejected pilot
as a speed result, and retain its raw evidence. Fixed GPU speculation keeps
its existing ADR-020 numerical-tolerance contract.

## Consequences

- Committed tokens per target pass equal `1 +` the mean accepted run length.
  The speed gain depends on acceptance, which is reported with every result
  and never assumed.
- Memory: the draft adds its own weights (644 MiB for Qwen2.5-0.5B Q8_0) and
  KV.
- Tests:
  - `SpeculativeOutputEqualsTargetGreedy`: managed, with a disagreeing and an
    identical draft;
  - `SpeculationRejectsIncompatibleModels` and
    `SpeculativeMatchesTargetOnMetal`;
  - `QwenHalfBillionSharesTheSevenBillionVocabulary`, which is
    `not_run_missing_model` without the 7B;
  - `CliSpeculativeGenerateKeepsTheTargetTokens`.

## Measured (2026-09-29)

- **Chat text.** On a chat prompt, 3 draft tokens raise Qwen2.5-7B-Instruct-1M decode from 17.4–18.3 to
  22.5–23.5 tokens/s. Acceptance is 54%, with 2.63 tokens per target pass.
- **Longer drafts lose.** 5 and 7 draft tokens are slower than dense, because a 6–8-token check costs about
  three single-token passes. The default is 3.
  - To lower that cost, Metal's matrix-vector threadgroups hold 4 and 8 rows for 2 and 4 tokens.
  - Runs of five or more tokens use the GEMM.
  - A small-batch kernel that keeps a 5–8-token check near one pass is future work.
- **Code-heavy text.** On the 3,528-token repository-summary prompt acceptance is 27–46%, and speculation is
  slower than dense.
