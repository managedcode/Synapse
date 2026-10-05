# Speculative decoding

Decision: ADR-020. Direction: FlyBrain F4.1–F4.2 in `flybrain.plan.md` (fewer target passes per committed token, same
output).

## Requirements

- `REQ-SPC-001`: `SpeculativeDecoding.Generate(target, draft, prompt, maxTokens, draftTokens)` returns the
  target's greedy continuation.
  - The draft proposes up to `draftTokens` tokens per round. The target verifies them in one batched step and
    commits the agreeing prefix plus its own next token.
  - Output is bitwise the target's greedy output on CPU, and equal within the decode tolerance on Metal.
- `REQ-SPC-002`: Incompatible pairs fail before any work:
  - token IDs that differ, checked by a SHA-256 over the draft vocabulary, types, and merges;
  - a draft vocabulary larger than the target's;
  - a target without batched steps;
  - `draftTokens + 1` above the target's logits rows;
  - the same instance passed as target and draft.
- `REQ-SPC-003`: `synapse generate --draft-model <gguf>` or `--draft-drop-layers i,j` runs speculatively and
  reports target passes, drafted and accepted tokens, and the acceptance rate. With only
  `--draft-drop-layers`, the draft is a layer-dropped copy of the target file (ADR-019).
- `REQ-SPC-005`: opt-in `adaptiveDepth: true` selects a depth from zero to
  `draftTokens` (at most seven on CPU, three on GPU) from measured committed tokens per millisecond.
  Wider adaptive GPU windows fail explicitly because the real Metal pilot
  lost greedy parity on the GEMM verification path.
  The CLI uses `--draft-tokens auto` with a cap of three. Zero runs ordinary
  target decode; every token remains target-verified. Result
  `DepthMeasurements` / CLI `depth_measurements` report per-depth actual rounds,
  accepted/committed counts, draft time including catch-up and verification time.
  Two initial samples per depth precede exploitation; a draft must beat plain
  throughput by 5%. Rotating reprobes back off from 16 to 128 rounds. Request
  tails are reported but excluded from the estimator. Timing noise, workload
  drift and exploration costs mean a speed improvement is not guaranteed.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-SPC-001-1` speculative output equals the target's greedy output with a disagreeing and an identical draft; an identical draft takes the minimum passes | `TEST-SPC-001-1` `SpeculativeOutputEqualsTargetGreedy` |
| `AC-SPC-001-2` the same holds on Metal at head dimension 128 | `TEST-SPC-001-2` `SpeculativeMatchesTargetOnMetal` |
| `AC-SPC-002-1` incompatible pairs and invalid draft lengths fail explicitly | `TEST-SPC-002-1` `SpeculationRejectsIncompatibleModels` |
| `AC-SPC-002-2` Qwen2.5-0.5B shares Qwen2.5-7B-Instruct-1M's token IDs | `TEST-SPC-002-2` `QwenHalfBillionSharesTheSevenBillionVocabulary` (`not_run_missing_model` without the 7B) |
| `AC-SPC-003-1` the CLI keeps the target's tokens and reports the draft | `TEST-SPC-003-1` `CliSpeculativeGenerateKeepsTheTargetTokens` |
| `AC-SPC-005-1` measured throughput selects profitable depth, permits zero, reprobes and rejects invalid costs | `TEST-SPC-005-1` `AdaptiveDepthTests` |
| `AC-SPC-005-2` adaptive CPU/Metal decode preserves greedy tokens through plain/draft transitions, accounts for all rounds and rejects unqualified GPU windows | `TEST-SPC-005-2` `AdaptiveOutputEqualsTargetGreedy`, `AdaptiveShortRequestsRemainExact`, `AdaptiveMetalRealModelKeepsGreedyTokenOrder`, `AdaptiveMetalRejectsUnqualifiedWideWindows` |
| `AC-SPC-005-3` CLI auto emits exact tokens and bounded measurements; fixed options remain valid | `TEST-SPC-005-3` `CliSpeculativeGenerateKeepsTheTargetTokens` |

## Evidence

Run X in `benchmarks/README.md`. Qwen2.5-7B-Instruct-1M Q8_0 target, Qwen2.5-0.5B Q8_0 draft, Metal, a
37-token chat prompt, 256 tokens:

| Draft tokens | Decode tok/s, round 1 / round 2 | Acceptance | Tokens per 7B pass |
|---:|---:|---:|---:|
| 0 (dense) | 17.4 / 18.3 | — | 1 |
| 3 | 23.5 / 22.5 | 54% | 2.63 |
| 5 | 14.0 / 13.5 | 40% | 3.00 |
| 7 | 14.2 / 13.9 | 32% | 3.27 |

- The default is 3 draft tokens. A 4-token verification step costs about 1.7 single-token passes; 6–8 tokens
  cost about 3, so longer drafts lose.
- On the repository-summary prompt the 0.5B agrees less, and speculation does not pay off: acceptance was 27–46%.

## Limitations

- The gain depends on how often the draft agrees with the target, which depends on the text. It is measured,
  never assumed.
- Verification steps of 5–8 tokens need a small-batch kernel whose cost stays near one pass.
- Auto is an online external-draft experiment. Startup GPU calibration,
  trained MTP heads and per-proposal confidence stopping are not implemented.
  Its exploration and draft-prefill cost is part of the request's wall time.
