# Layer drop

Decision: ADR-019. Semantics: ADR-002 and ADR-003 (`ApproximateProfile`, `BypassOutputs`). Plan:
`gpu-kernels.plan.md` (DROP.1).

## Requirements

- `REQ-LDP-001`: `ModelLoadOptions.LayerDrop` drops whole source layers. Every backend runs the kept layers as
  a shallower model: the residual passes through, and the dropped layers' weights and KV are never touched.
  - The result equals the model built from the kept layers only: bitwise on CPU and on Metal.
  - KV is sized for the kept layers.
- `REQ-LDP-002`: A drop is an approximation profile.
  - Without evidence it is experimental: the runtime profile gains `+drop<n>x`.
  - With an evidence digest it is quality-bounded: `+drop<n>`.
  - The graph marks each dropped layer as a `ProfileDecision` with `ApproximateProvenance` and a bypass, so
    the fingerprint changes.
  - Invalid sets fail at load: empty, duplicate, out of range, or every layer.
- `REQ-LDP-003`: A dropped layer's pages are never loaded.
  - CPU prefetch skips its tensors.
  - Metal wraps only the page-aligned segments the plan reads, so the GPU never makes the dropped ranges
    resident.
- `REQ-LDP-004`: The CLI accepts `--drop-layers i,j` (experimental) and `--drop-profile <evidence.json>`
  (qualified: the layers come from `qualifiedDrop.layers` and the file's SHA-256 is the evidence) on `generate`
  and `score`.

## Acceptance criteria and tests

| Criterion | Test |
|---|---|
| `AC-LDP-001-1` a drop equals the shallower model on the reference, managed, and Metal backends, with the same KV bytes | `TEST-LDP-001-1` `DroppedLayersEqualAShallowerModel` |
| `AC-LDP-001-2` KV bytes scale with the kept layers | `TEST-LDP-001-2` `LayerDropSizesKvForKeptLayers` |
| `AC-LDP-002-1` profiles are named, experimental versus qualified, and invalid sets fail | `TEST-LDP-002-1` `LayerDropIsNamedAndValidated` |
| `AC-LDP-002-2` dropped regions are profile bypasses in a graph that still verifies | `TEST-LDP-002-2` `DroppedRegionsAreProfileBypassesInTheGraph` |
| `AC-LDP-003-1` prefetch excludes exactly the dropped tensors | `TEST-LDP-003-1` `LayerDropNeverPrefetchesDroppedWeights` |
| `AC-LDP-003-2` weight segments leave dropped ranges unmapped and merge units that share pages | `TEST-LDP-003-2` `weight_segments_leave_dropped_ranges_unmapped`, `weight_segments_merge_units_that_share_pages` (Rust) |
| `AC-LDP-004-1` the CLI runs experimental and qualified drops and rejects dropping every layer | `TEST-LDP-004-1` `CliGenerateAndScoreAcceptLayerDrops` |

## Evidence

- `benchmarks/results/2026-09-29-m2-pro-qwen2.5-7b-1m-q8_0-layer-drop-qualification.json`:
  - Qwen2.5-7B-Instruct-1M, 1,024 scored positions of the pinned haystack; dense perplexity 12.163.
  - Layers 0–3 and 27 are essential.
  - Dropping {11, 12} changes perplexity by −1.1% with 79.8% top-1 agreement. This is the qualified set, and
    its scope is this text's perplexity only.
  - Three layers cost +11.7%, four +39.2%, eight +137%.
- `benchmarks/results/2026-09-29-m2-pro-qwen2.5-0.5b-q8_0-layer-drop-qualification.json`: no layer of the
  0.5B is cheap. The best single drop costs +7.6% perplexity. No set is qualified.
- Memory on Metal (7B Q8_0, 3,528-token prompt, peak RSS; in run W of `benchmarks/README.md`):

  | Layers dropped | Peak RSS | Change |
  |---:|---:|---:|
  | 0 | 7,255 MiB | — |
  | 2 | 6,780 MiB | −475 MiB |
  | 4 | 6,326 MiB | −929 MiB |
  | 8 | 5,381 MiB | −1,874 MiB |

  The 4- and 8-layer rows are experimental runs that show the memory effect; they are not qualified.

## Limitations

- An unhealed checkpoint loses quality quickly. Quantization saves more memory per quality point than dropping
  layers: the 7B in Q4_K_M peaks at 4,233 MiB in Synapse, with perplexity 11.93 against 12.16 for Q8_0
  (ADR-021).
- CUDA copies the whole file to the device. It does not segment weights yet.
