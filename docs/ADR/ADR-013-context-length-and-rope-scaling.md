# ADR-013: Context length limits and explicit RoPE scaling

Status: Accepted. Date: 2026-09-28.

## Context

The owner asked what happens at 30k, 40k, and 128k tokens of context. The
pinned Qwen2.5-0.5B-Instruct GGUF declares `qwen2.context_length = 32768` and
no rope-scaling keys. Before this decision, `Qwen2ModelComposition` set the
instance context to `min(requested, 32768)`. A request for 40,960 or 131,072
positions was therefore silently clamped, and the first longer prompt failed
later with a generic `ArgumentOutOfRangeException`.

Qwen publishes YaRN as the method for running Qwen2.5 beyond its trained
window (`rope_scaling = {type: yarn, factor: 4, original_max_position_embeddings:
32768}`), and llama.cpp implements the same formula (`rope_yarn` in ggml).
YaRN changes the model math at every position, including short prompts, so it
cannot be an implicit fallback.

## Decision

- **No silent clamp.** A requested `ModelLoadOptions.ContextSize` above the
  model's trained context fails at load with `NotSupportedException` naming
  both numbers, unless an explicit RoPE scaling profile covers it.
- **Explicit scaling profile.** `ModelLoadOptions.RopeScaling` is `none`
  (default) or `yarn` with a factor and the original trained context. The
  extended limit is `floor(factor × original)`. A GGUF that declares
  `<arch>.rope.scaling.*` keys supplies the profile; an explicit option that
  contradicts the file fails.
- **One frequency source.** `RopeFrequencies` computes every cosine and sine
  once, in C#, for all backends. With `none` it keeps the existing FP32
  expression `position / MathF.Pow(theta, 2i/d)` bitwise. With `yarn` it
  follows ggml `rope_yarn`: `theta_interp = theta_extrap / factor`, a linear
  ramp between the correction dimensions for `beta_fast = 32` and
  `beta_slow = 1`, `ext_factor = 1`, and the magnitude scale
  `mscale = 1 + 0.1 ln(factor)` applied to both cosine and sine. The reference
  backend uses the same helper, so the oracle and the optimized paths stay
  comparable.
- **Model IR carries the profile.** `RopeAttributes` gains an optional
  `RopeScaling` record, so the canonical graph fingerprint distinguishes a
  scaled instance from an unscaled one, and the IR context bound becomes the
  extended limit.
- **Runtime profile names the scaling.** A scaled instance appends
  `+yarn<factor>` to its runtime profile, and the CLI accepts
  `--rope-scaling yarn:<factor>`.
- **Long prompts are data, not arguments.** The CLI accepts `--tokens-file`
  for prompts too long for `argv`, and it reports prefill progress on standard
  error during long prompt evaluation.

## Consequences

- 32,768 tokens and below behave exactly as before for unscaled models.
- 40,960 and 131,072 tokens require `--rope-scaling yarn:<factor>` on this
  model. Quality at those lengths is an empirical question: the long-context
  evaluation reports retrieval results instead of assuming them.
- KV memory stays proportional to the instance context: FP32 KV for
  Qwen2.5-0.5B is 24 KiB per position per slot (768 MiB at 32k, 3 GiB at
  128k). Callers size `MaximumConcurrentSessions` accordingly.
- Tests: `ContextBeyondTrainedLengthFailsExplicitly`,
  `YarnFrequenciesMatchFp64Formula`, `UnscaledFrequenciesStayBitwise`,
  `YarnContextExtendsLimitExactly`, `ContradictingScalingFails`, and the CLI
  `--tokens-file` and progress tests in `docs/Features/LongContext.md`.
