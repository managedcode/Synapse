# ADR-019: Qualified layer drop

Status: Accepted. Date: 2026-09-29. Implements the `ApproximateProfile`
eligibility of ADR-002 and ADR-003 for whole dense layers.

## Context

The owner's direction is a model that is not loaded whole: parts are dropped,
and those parts cost neither memory nor time.

- **The rule.** ADR-002 forbids arbitrary skips of a dense checkpoint's
  layers, because a skip changes the model's function. Skipping is allowed
  only in an approximation profile tied to quality evidence. `AGENTS.md`
  says the same: no dense-layer skip without a qualified numerical profile.
- **The literature.** ShortGPT (arXiv 2403.03853) and Gromov et al. (arXiv
  2403.17887) show that deep decoders have redundant middle and late layers.
  Removing a quarter of them costs a measurable but bounded quality loss,
  which is larger for retrieval and long context. The loss must therefore be
  measured per model and per task, never assumed.
- **Where the gain is.** Decode is bandwidth-bound, so every dropped layer
  removes its weight bytes from every token. A dropped layer also needs no KV
  cache. The weights are read through a read-only memory map (ADR-012), so
  pages that are never touched are never loaded: the model is not loaded
  whole.

## Decision

- **Profile.** `ModelLoadOptions.LayerDrop` is a `LayerDropProfile`:
  - sorted, distinct source layer indices;
  - an optional `EvidenceSha256`, the SHA-256 of the evaluation evidence file
    that qualified the profile.

  At least one layer must stay. Indices must lie inside the model.
- **Numerical mode.** A profile with evidence is `QualityBounded`. Without
  evidence it is `Experimental`: the measuring runs that produce the evidence.
  - The runtime profile gains `+drop<count>`, plus `x` when experimental, so
    evidence can never be confused with the full model.
  - The dense path stays the default.
- **Semantics.** A dropped layer is a session-wide `ProfileDecision` with
  `ApproximateProvenance` and a `BypassOutputs` map from the block output to
  its hidden input, so the residual stream passes through unchanged.
  - Its K/V state is never written and never read, because its only reader
    is its own attention.
  - The verifier therefore exempts a profile-decided region whose state slots
    are read only inside that region from the position-hole rule. The
    exemption is sound because a session-wide profile never runs the region
    for some positions and not others.
  - Experimental graphs key the provenance with the SHA-256 of the canonical
    request (`experimental-layer-drop:<layers>`), so the fingerprint still
    differs from the dense model and from every other drop set.
- **Execution.** Every backend runs the kept layers as a shallower model:
  - the executors receive only the kept layers' weights and offsets;
  - KV slots are sized for the kept layers only;
  - on Metal no ABI change is needed, because the descriptor carries the
    kept layer count and their offsets.
- **Qualification.** A profile is chosen from measured per-layer perplexity
  deltas on the pinned haystack. Its evidence reports three axes against the
  dense model on identical tokens:
  - quality: perplexity, the long-context task suite, and greedy agreement;
  - memory: resident weights and KV bytes;
  - speed: time to first token and decode tokens per second.

## Consequences

- Dropping `k` of `L` layers removes about `k/L` of the layer weights from
  memory and from every token, and `k/L` of the KV cache.
- Quality loss is reported, never hidden. A profile qualified on one task is
  not qualified for another.
- The dense Model IR function is unchanged. A drop set is an explicit profile
  in the graph, and the graph fingerprint records it.
- Tests:
  - `DroppedLayersEqualAShallowerModel`: bitwise on the reference, managed,
    and Metal backends;
  - `LayerDropSizesKvForKeptLayers`, `LayerDropIsNamedAndValidated`, and
    `DroppedRegionsAreProfileBypassesInTheGraph`;
  - `LayerDropNeverPrefetchesDroppedWeights` and
    `CliGenerateAndScoreAcceptLayerDrops`;
  - the Rust tests `weight_segments_leave_dropped_ranges_unmapped` and
    `weight_segments_merge_units_that_share_pages`.

## Implementation notes (2026-09-29)

- **Metal needs weight segments.** A no-copy buffer over the whole mapped file keeps every page resident while
  it is bound. A 2-layer drop still peaked at 7,271 MiB until the decoder wrapped only the page-aligned
  segments the plan reads (`DecoderPlan::weight_segments`) and bound each launch's segment. After that the 7B
  peaks at 7,255 MiB dense, 6,780 (2 layers dropped), 6,326 (4), and 5,381 (8).
- **Qualification.** On Qwen2.5-7B-Instruct-1M, {11, 12} is the qualified set: −1.1% perplexity with 79.8%
  top-1 agreement. The evidence file's SHA-256 is its provenance. Qwen2.5-0.5B has no cheap layer, so no set is
  qualified for it.
- **CUDA.** CUDA still copies the whole file to the device as one segment.
