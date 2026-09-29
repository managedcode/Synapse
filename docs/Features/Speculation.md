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

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-SPC-001-1` speculative output equals the target's greedy output with a disagreeing and an identical draft; an identical draft takes the minimum passes | `TEST-SPC-001-1` `SpeculativeOutputEqualsTargetGreedy` |
| `AC-SPC-001-2` the same holds on Metal at head dimension 128 | `TEST-SPC-001-2` `SpeculativeMatchesTargetOnMetal` |
| `AC-SPC-002-1` incompatible pairs and invalid draft lengths fail explicitly | `TEST-SPC-002-1` `SpeculationRejectsIncompatibleModels` |
| `AC-SPC-002-2` Qwen2.5-0.5B shares Qwen2.5-7B-Instruct-1M's token IDs | `TEST-SPC-002-2` `QwenHalfBillionSharesTheSevenBillionVocabulary` (`not_run_missing_model` without the 7B) |
| `AC-SPC-003-1` the CLI keeps the target's tokens and reports the draft | `TEST-SPC-003-1` `CliSpeculativeGenerateKeepsTheTargetTokens` |

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
