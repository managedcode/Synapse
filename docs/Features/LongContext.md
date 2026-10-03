# Long context

## Controlled long-prompt ablation (`REQ-CTX-009`)

ADR-025 adds explicit CLI modes `--optimization off|dense|custom` and an
`optimization-eval` plan for prepared `.synapse` artifacts. `off` selects the
scalar reference path with FP32 KV, full layers and dense attention. `dense`
keeps the requested backend while disabling approximations. `custom` permits
explicit supported KV/page/layer settings.
Contradictory options fail. These controls describe executed math; they do not
assign a quality rating to a profile.

The evaluator pairs identical prompt IDs across profiles and records real
child-process generation, bounded-tail teacher-forced scoring, answer grades,
output lengths, TTFT, generation/decode time, actual KV allocation and observed
process memory. Original weights with FP32 KV, full layers, dense attention and
reuse disabled are the baseline. Cold and repeated prompts have separate rows;
the repeated no-reuse baseline is required to measure prefix reuse. Aligned
greedy agreement and NLL drift diagnose numerical effects; tail perplexity on
retrieval prompts is not held-out model qualification.
Paired profiles use the same parsed backend as their named dense baseline.
Scores retain finite mean NLL and log perplexity. The exponentiated tail
perplexity is nullable and carries `finite`, `overflow` or `underflow` status;
paired ratios use log differences and carry the same range status. Nonfinite
scores fail before entering evidence, preserving earlier generation phases.

`TASK-CTX-009` remains in progress. `AC/TEST-CTX-009-1` maps CLI controls,
`-2` maps strict plans and prepared-only execution, `-3` maps real generation,
scoring and provenance, `-4` maps repeat controls, and `-5` maps paired ordering,
failure/cancellation evidence and absence of implicit conversion/promotion.
Raw measurements, exact verification commands and limitations are recorded in
the dated development evidence after execution.

## Dynamic memory workload (`REQ-CTX-010`)

`memory-eval --request <json> --output <new.json>` executes a prepared Qwen2
package through a real local model. Its version-one request contains
`modelPath`, `contextSize`, bounded `shortPromptTokens` and `longPromptTokens`,
`maximumNewTokens`, `threads`, an explicit backend/KV/page/layer/reuse `profile`,
optional RoPE, and `scoredTailTokens` (zero disables scoring). The long input
must exceed the short input, and its prompt plus output must fit the context.
Paths are relative to the request file; `profile.modelPath` is rejected because
the request supplies the model path. Unknown properties and unsupported
profiles fail; execution never converts a source model.

The main model runs `loaded`, `short-first`, `long`, `short-after-long`, then
`disposed`. Separate cold models run `fresh-short` and `fresh-long` with identical
IDs and the same numerical profile. Generation rows snapshot actual allocated
KV, current working set, available macOS physical footprint, and managed heap
before optional scoring. Every scoring row has its own `-score` label because
scoring can reserve and contract KV too. Each row records its actual runtime,
kernel and graph identities. The package/source/file identity comes from an
actual untimed inspection and hash before load timing.

Fresh-model comparisons retain output-ID equality and, when scoring is enabled,
aligned greedy agreement and NLL drift. These measure the numerical effect of
capacity changes, not held-out model quality. Five-millisecond process sampling
records peaks separately from instantaneous checkpoints; no forced collection
or file-cache flush occurs. Zero owned KV after disposal does not promise an
immediate drop in process RSS. Partial phases are published atomically and
retained on failure or cancellation; existing output files are preserved.
`AC/TEST-CTX-010-4` covers real phases, provenance, quality, bounded failures,
cancellation and publication. Actual speed/memory claims remain pending paired
raw measurements and the corresponding quality result.

Decisions are in ADR-013 (context limits and RoPE scaling), ADR-015 (quality
evaluation), and ADR-012 (GPU execution). The plan is `gpu-kernels.plan.md`.

## Requirements

- `REQ-CTX-001`: A context request above the model's trained window fails at
  load with a message naming both numbers. Nothing is clamped silently.
- `REQ-CTX-002`: An explicit YaRN profile extends the window to
  `floor(factor × original)`. Every backend reads the same C#-computed RoPE
  frequencies; the unscaled path stays bitwise unchanged; the Model IR and the
  runtime profile carry the profile.
- `REQ-CTX-003`: Long prompts enter the CLI through `--tokens-file`, prefill
  progress is reported during evaluation, and 32k, 40k, and 128k runs record
  raw timings, memory, and a retrieval result as diagnostic evidence.
- `REQ-CTX-004`: Teacher-forced scoring (ADR-015) returns the per-position
  negative log-likelihood and greedy token on every backend. `synapse score`
  reports perplexity with the llama-perplexity protocol, so Synapse and
  llama.cpp can be compared on identical tokens at every context length.
- `REQ-CTX-005`: Long-context answer quality is measured, not assumed.
  Deterministic exact-answer tasks (single needle, multi-key needle with
  distractors, variable tracking) run at several lengths and depths. Every
  engine gets the same prompt, and each engine's token identity is recorded
  next to its answer.
- `REQ-CTX-006`: Query-aware KV page activation (ADR-016) is an explicit, opt-in
  approximation profile. A decode token attends to the sink page, the recent
  window, and a budget of pages ranked by a key upper bound. A budget that
  covers the prefix equals dense attention bitwise. A seeded random selection
  at the same budget is the control. Backends that do not implement it fail
  at load.
- `REQ-CTX-007`: KV memory follows the context in use (ADR-017). A slot grows on
  demand up to the instance context. Growth never changes the numbers, and the
  allocated KV bytes are observable.
- `REQ-CTX-008`: With `ReusePromptPrefix`, the direct session reuses the K and V
  of the longest token prefix it already holds (ADR-018). It evaluates only the
  new tokens and reports how many were reused. Any other direct use
  invalidates the remembered tokens.
- `REQ-CTX-010`: Known requests contract oversized KV allocations when their
  rounded need is at most a quarter of existing capacity, preserving surviving
  bytes and prefix reuse (ADR-017 amendment). Each newly assigned batch slot is
  reserved under the execution gate. Disposal releases CPU and native KV;
  allocation observations distinguish owned KV from process memory.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-CTX-001-1` a context above the trained window fails at load and names the requested and trained sizes; the trained size itself loads | `TEST-CTX-001-1` `ContextBeyondTrainedLengthFailsExplicitly` |
| `AC-CTX-002-1` YaRN cosines and sines match an FP64 evaluation of the ggml formula | `TEST-CTX-002-1` `YarnFrequenciesMatchFp64Formula` |
| `AC-CTX-002-2` without scaling the table equals the previous FP32 expression bitwise | `TEST-CTX-002-2` `UnscaledFrequenciesStayBitwise` |
| `AC-CTX-002-3` a YaRN instance accepts exactly `factor × original` positions, reports the profile, and its graph fingerprint differs from the unscaled graph | `TEST-CTX-002-3` `YarnContextExtendsLimitExactly` |
| `AC-CTX-002-4` a file-declared profile is used, and an explicit option that contradicts it fails | `TEST-CTX-002-4` `ContradictingScalingFails` |
| `AC-CTX-003-1` `--tokens-file` feeds the same tokens as `--tokens`, reports progress on standard error, and rejects a malformed file | `TEST-CTX-003-1` `CliTokensFileMatchesInlineTokens`, `CliRejectsMalformedTokensFile` |
| `AC-CTX-003-2` the pass-key diagnostic builds prompts from the locked token fixture, runs the real CLI per prompt, records the answer check, and rejects prompts longer than the context | `TEST-CTX-003-2` `PasskeyCommandFindsKeyInShortPrompt`, `PasskeyCommandRejectsPromptsLongerThanContext` |
| `AC-CTX-004-1` scored NLL and greedy tokens equal independent prefills on the reference, managed, and native backends (NLL within 0.02) | `TEST-CTX-004-1` `ScoreMatchesIndependentPrefills` |
| `AC-CTX-004-2` native Q8 activations track the FP32 reference: identical greedy tokens, mean NLL within 0.02; the model completes the capital of Ukraine | `TEST-CTX-004-2` `ScoreIsBackendConsistentAndKnowsCapitals` |
| `AC-CTX-004-3` Metal scoring equals the reference greedy tokens with per-position NLL within 0.01 at 3 and 64 rows per step | `TEST-CTX-004-3` `MetalScoreTracksReference` |
| `AC-CTX-004-4` invalid scoring rows, ranges, and over-long inputs fail; progress reaches every evaluated position | `TEST-CTX-004-4` `ScoringArgumentsAreValidated`, `ScoreReportsProgressThroughEveryPosition` |
| `AC-CTX-004-5` `synapse score` reports chunked perplexity, the token-ID hash, and a per-position trace; it rejects input shorter than one chunk; `generate` reports decoded text | `TEST-CTX-004-5` `CliScoreReportsPerplexityWithTheLlamaProtocol`, `CliScoreRejectsInputShorterThanOneChunk`, `CliGenerateReportsDecodedText` |
| `AC-CTX-005-1` quality cases are deterministic per seed, stay within the token budget, contain their answers, and respect needle depth; a short haystack fails explicitly | `TEST-CTX-005-1` `CasesAreDeterministicAndHoldTheirAnswers`, `DepthControlsNeedlePlacement`, `HaystackShorterThanTheBudgetFailsExplicitly` |
| `AC-CTX-006-1` the page bound never underestimates a query-key product; selection keeps the sink, the window, and the best page; the random control is deterministic and honours the budget | `TEST-CTX-006-1` `KeyBoundNeverUnderestimatesAPageScore`, `SelectionKeepsSinkWindowAndTheBestPage`, `RandomControlSelectsTheBudgetDeterministically` |
| `AC-CTX-006-2` a covering budget equals dense logits bitwise, and a small budget changes them | `TEST-CTX-006-2` `KvPagesCoveringThePrefixEqualDense` |
| `AC-CTX-006-3` the profile is named in the runtime profile and rejected with ADR-016 on the reference and Metal backends | `TEST-CTX-006-3` `KvPagesAreNamedAndRejectedWhereNotImplemented` |
| `AC-CTX-006-4` prompt tokens attend densely even alone in a step (a 641-token prompt in 64-token chunks, a one-token reused tail) | `TEST-CTX-006-4` `PromptTokensStayDenseEvenAloneInAStep` |
| `AC-CTX-007-1` the CPU cache grows by unit and doubling, keeps its contents, and rejects positions at the context | `TEST-CTX-007-1` `CpuKvCacheGrowsAndKeepsContents` |
| `AC-CTX-007-2` a slot that grows many times gives bitwise the logits of a full-size slot on Metal and CPU, with fewer bytes | `TEST-CTX-007-2` `GrowingKvSlotMatchesFullSlotBitwise` |
| `AC-CTX-007-3` allocated KV bytes follow the positions in use (1,024 then 2,048 positions) | `TEST-CTX-007-3` `KvBytesFollowTheContextInUse`, Rust `kv_capacity_grows_by_unit_and_doubling_up_to_the_context` |
| `AC-CTX-007-4` a request of known size reserves its slot once (3,072 positions, not 4,096 by growth) | `TEST-CTX-007-4` `GenerateReservesPromptAndOutputOnce` |
| `AC-CTX-008-1` a reused prefix is skipped and the answer equals a cold run: bitwise on CPU, the same greedy tokens on Metal | `TEST-CTX-008-1` `ReusedPrefixSkipsSharedTokensAndMatchesAColdRunOnCpu`, `ReusedPrefixMatchesAColdRunOnMetal` |
| `AC-CTX-008-2` an identical prompt re-evaluates only its last token, and other direct work invalidates the reuse | `TEST-CTX-008-2` `IdenticalPromptReevaluatesOnlyItsLastTokenAndOtherDirectWorkInvalidates` |
| `AC-CTX-008-3` a request that fails part way keeps only the prefix it never overwrote, so a retry reuses no stale K or V | `TEST-CTX-008-3` `FailedRequestKeepsOnlyThePrefixItDidNotOverwrite` |

## Current evidence

Local diagnostics on the owner M2 Pro (19-core GPU, 32 GB), one process per
row. They carry no paired or statistical claim.

- **Limits.**
  - Qwen2.5-0.5B declares 32,768 trained positions.
  - A request for 32,769, 40,960, or 131,072 positions without scaling now
    fails at load (`ContextBeyondTrainedLengthFailsExplicitly`).
  - `--rope-scaling yarn:4:32768` allows 131,072 positions.
- **Pass-key retrieval, native window** (Metal, FP32 KV): 9 of 9 exact answers
  at 4,095, 16,383, and 31,983 prompt tokens with the key at 10%, 50%, and 90%
  depth. Prefill took 1.7 s, 12.4 s, and 39.2 s.
  Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-passkey-native-f32-diagnostic.json`.
- **Pass-key retrieval beyond the trained window** (Metal, YaRN ×4, FP16 KV, context 131,072):
  - At 65,535 prompt tokens: 3 of 3 exact.
  - At 119,991 tokens: exact at 50% and 90% depth. At 10% depth the model answered only the first digit
    (`6`, then end of turn) instead of `68317`.
  - Prefill took 153 s at 65k and 585 s at 120k. Decode ran at 50–55 tok/s at 120k.
  - Peak footprint was 1.8 GB, against 3.4 GB with FP32 KV.
  - Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-passkey-yarn4-f16-diagnostic.json`.
- **llama.cpp control** (Metal, same YaRN ×4, FP16 KV, key at 10% depth). **Corrected:** it was
  recorded as using identical 119,991 token IDs, but llama.cpp `-f` dropped the final newline, so
  it evaluated 119,990 tokens (see the correction below).
  - The first run missed the key and printed filler text. The rerun on truly identical 119,991
    tokens answered exactly like Synapse: `6`, then end of turn. Both engines agree, so the miss is
    the model under YaRN ×4.
  - Prefill: llama.cpp 288 s (417 tok/s) against Synapse 585 s (205 tok/s). Long-context prefill
    attention is the open gap (`gpu-kernels.plan.md` GPU.10).
  - Decode: 48 tok/s against 50–55 tok/s. Peak footprint: 1.95 GB against 1.80 GB.
  - Raw data: `benchmarks/results/2026-09-28-m2-pro-llamacpp-metal-passkey-120k-control-diagnostic.json`.
- **Speed at 40,000 and 131,000 synthetic tokens** (YaRN ×4, FP32 KV):
  - Prefill averaged 670 and 151 tok/s; decode ran at 67 and 33 tok/s.
  - Peak footprint was about 3.4 GB, dominated by the 3 GiB KV slot sized for
    131,072 positions.
  - Synthetic token IDs make the generated text meaningless; retrieval quality
    comes only from the pass-key runs.
  - Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-long-context-speed-diagnostic.json`.
- **CPU.** The CPU backends keep FP32 KV and scalar-order attention, so a
  32k prefill takes minutes. No CPU long-context timing is recorded yet.

## Quality evidence (ADR-015), up to the trained 32k window

Local diagnostics on the owner M2 Pro. The owner capped this round at 32k.
YaRN lengths are not part of it.

**Perplexity parity on identical tokens.** Corpus: `corpus-all`, built from
`b090e95` (255,478 tokens). Its token IDs equal `llama-tokenize`'s.

| Context | llama.cpp | Synapse FP32 KV | Synapse FP16 KV |
|---|---:|---:|---:|
| 4,096 (CPU, 2 chunks) | 11.3889 | 11.3897 (native) | — |
| 4,096 (Metal, 16 chunks) | 12.7728 | 12.7724 | 12.7723 |
| 16,384 (Metal, 8 chunks) | 6.8868 | 6.8867 | 6.8868 |
| 32,768 (Metal, 7 chunks) | 3.9687 | 3.9686 | 3.9686 |

- Every per-chunk value agrees within 0.01%.
- On identical tokens, FP16 KV and FP32 KV pick the same greedy token at
  99.93% of positions, with a mean |ΔNLL| of 0.0012 nats.
- Perplexity falls from 12.8 at 4k to 4.0 at 32k, so the model uses the
  longer context.
- Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-perplexity-parity-diagnostic.json`.

**Exact-answer tasks.**
- **Setup.** 28 cases: needle and multi-key at depths 0.1, 0.5, and 0.9,
  and variable tracking, at 4k, 8k, 16k, and 32k targets. The haystack is
  the pinned repository documentation followed by the code.
- **Engine identity.** Synapse Metal FP32 KV, Synapse Metal FP16 KV, and
  llama.cpp Metal gave byte-identical answers in 28 of 28 cases, on prompt
  token IDs verified identical in every run.
- **MLX** (SwiftLM, separately converted 8-bit weights) agreed in 23 of 28
  and was never more correct.
- **CPU pair at 4k.** Synapse native (Rust Q8 kernels, FP32 KV) and llama.cpp
  CPU gave byte-identical answers in 7 of 7 cases on identical IDs. Raw data:
  `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-long-context-quality-cpu-diagnostic.json`.
- **Model limits.** The failures are shared by every engine, so they are
  limits of the 0.5B model:
  - single needle: 11 of 12 (it missed at 32k, 50% depth);
  - multi-key with three distractors: 9 of 12;
  - variable tracking: 0 of 4.
- **Raw data:** `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-long-context-quality-diagnostic.json`.
- **Limitation.** Wall times in that file were recorded while a CPU
  experiment ran in parallel, so they are not speed evidence. Speed comes
  from the context sweep on a quiet machine.

**Correction of the earlier 120k llama.cpp control.** llama.cpp `-f` drops
the prompt's trailing newline. The first control therefore evaluated 119,990
tokens (its own log says so), not the 119,991 recorded as identical, and it
printed filler text. The rerun used the prompt text plus one extra newline, so
llama.cpp reported 119,991 tokens. It answered `6`, then end of turn: the
same answer as Synapse. The evidence file keeps both runs.

## KV page activation, CPU quality evidence (ADR-016)

This is decode-mode perplexity (`score --scoring-rows 1`), so every scored
position goes through page selection. Runs are on the native CPU backend
over the pinned corpus. "Read" is the share of the context a decode step
attends to.

| Context | Selection | Budget | Read | Perplexity | Change | Greedy |
|---|---|---|---:|---:|---:|---:|
| 8k | dense | — | 100% | 8.936 | — | 59.0% |
| 8k | key bound, 16-token pages | 512 tokens | ~10% | 11.903 | +33% | 53.7% |
| 8k | random, 16-token pages | 512 tokens | ~10% | 21.132 | +136% | 45.5% |
| 8k | key bound, 64-token pages | 512 tokens | ~11% | 16.408 | +84% | 46.9% |
| 8k | key bound, 16-token pages | 2,048 tokens | ~30% | 9.236 | +3.4% | 58.4% |
| 8k | random, 16-token pages | 2,048 tokens | ~30% | 13.916 | +56% | 49.4% |
| 8k | key bound, 64-token pages | 2,048 tokens | ~31% | 10.484 | +17% | 55.7% |
| 16k | dense | — | 100% | 43.673 | — | 35.3% |
| 16k | key bound, 16-token pages | 512 tokens | ~5% | 52.498 | +20% | 33.9% |
| 16k | key bound, 16-token pages | 2,048 tokens | ~14% | 46.381 | +6.2% | 34.7% |
| 16k | random, 16-token pages | 2,048 tokens | ~14% | 58.522 | +34% | 32.4% |

- **Setup.** The window is 256 tokens for every run. The 8k and 16k rows score
  the last 1,023 positions of the first chunk.
- **Key-bound selection is real signal.** At the same budget, the random
  control loses 5–16 times more.
- **Smaller pages help.** 16-token pages bound the keys much more tightly
  than 64-token pages.
- **The needed share falls as context grows.** 14% of a 16k context costs
  6%, while 30% of an 8k context costs 3%.
- **It is not free.** The profile stays opt-in and approximate. The GPU
  kernels, and with them any speed or memory benefit, are not implemented or
  measured yet.
- Raw data: `benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-kv-page-activation-cpu-diagnostic.json`.
