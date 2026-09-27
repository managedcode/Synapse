# FlyBrain graph-native execution: brainstorm

Date: 2026-09-27. Status: planning input for `flybrain.plan.md`; nothing
here is implemented or measured.

## The idea, restated precisely

A model is a graph of coarse executable regions. Inference is an activation
wave through the subgraph that a request actually needs. Four independent axes
are optimized per request:

| Axis | Question | Synapse artifact |
|---|---|---|
| What | Which regions execute for this token/step/session? | Model IR regions + `RegionActivation` (ADR-003) |
| When resident | Which region weights must be in RAM now? | Weight residency manager + reservation ledger |
| Where | Which device/worker runs each region? | Deployment plan + direct data plane |
| Precision | Which encoding does each active region use? | Execution IR precision profile |

The goal is to optimize **how much of the model has to run and be resident**,
not only how fast one matmul is.

## What is established vs. hypothesis

Established, and usable now:

- Region scheduling, residency, and placement are engineering mechanisms. They
  can be built and verified on a dense checkpoint where every region is
  required. Paging then acts as a capacity path, and placement as a pipeline.
- A real MoE checkpoint's router is part of the model's function. Skipping its
  unselected experts is exact, and it is the natural first target for "region
  not resident" (expert paging).
- Speculative decoding verifies drafts with the full target model. Its greedy
  output matches the target's own greedy output, so the target does less work
  per committed token without changing the output. An external draft from the
  same tokenizer family (Qwen2.5-0.5B for Qwen2.5-Coder-1.5B) is the Apache
  path. Early-exit self-speculation drafts with the target's own first layers.
  It needs a checkpoint trained for early exit, and the published LayerSkip
  checkpoints are gated and FAIR-noncommercial, so they are local research
  only.
- Mixture-of-Depths confirms the state rule ADR-003 adopts: a token that skips
  a block contributes no keys or values at that block, and later tokens cannot
  attend to it there. Top-k over the sequence is non-causal, so decode needs a
  causal predictor.
- In decode, the value crossing a transformer layer boundary is one hidden
  vector per token: `hidden_size × 4` bytes in FP32, which is 3.5 KiB for
  Qwen2.5-0.5B (hidden 896). The residual stream is already a small "latent
  edge". A layer-range pipeline ships that vector plus the position, and each
  region owner keeps its own KV. JEPA is not required for small inter-region
  traffic; it stays a separate research line (R06).

Hypotheses that need evidence before any claim:

- "CSharp", ".NET", and "Math" regions as separable weights. Dense
  checkpoints store knowledge in superposition; no disjoint "C# weights" exist
  to page. The honest path is to measure first, then train:
  1. Route and expert histograms per corpus on a real MoE (C#, other code,
     prose, math) from activation traces.
  2. Domain experts trained separately and merged into an MoE (Branch-Train-MiX
     family), or a router tuned on a dense model. Both need native training.
- Per-token dynamic depth on an existing dense checkpoint (Mixture-of-Depths,
  Router-Tuning). This requires a trained router and makes the result
  `QualityBounded` with full/random/static/learned controls.
- A small resident working set for a large model. The union of experts touched
  across tokens and sessions can approach the whole model. We measure bytes
  touched, resident, loaded, and transferred separately.
- Region-conditioned precision (C# region at higher precision) requires a
  sealed C# quality suite and equal-memory controls (spec R02/G11).

## Decisions proposed

1. Make the Model IR the thing that runs before adding any skipping. Today it
   is built and verified, but the Qwen2 executor ignores it.
2. Split eligibility into decision, provenance, and skip semantics (ADR-003).
   Provenance, not labels, decides the numerical mode.
3. Two regions per transformer layer (attention, MLP). They have different
   state consequences (spec §12.3), and they are the units that MoE, dynamic
   depth, precision, and paging act on.
4. Order the first non-dense work by legitimacy: tiny conditional fixture,
   then LayerSkip self-speculation (exact), then real MoE expert routing and
   paging (exact), then learned routing (quality-bounded).
5. Orleans holds placement, leases, and epochs for region ranges. The wave
   itself travels over the worker data plane. There are no grain calls per
   token or region.
6. Region-level batching: tokens from different sessions that reach the same
   region (or expert) can share one kernel launch. The FlyBrain scheduler is
   the natural home for continuous batching.

## Rejected

- Skipping layers of an unchanged dense checkpoint because a label or
  heuristic says the request is "C#".
- One grain per region activation, token, or neuron.
- Shipping tensors through Orleans messages.
- A verification-only IR beside a hand-written executor.
- Using published speedups (MoD, LayerSkip, Router-Tuning) as Synapse
  targets or results.
