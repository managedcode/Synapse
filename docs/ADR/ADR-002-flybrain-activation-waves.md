# ADR-002: FlyBrain activation waves and coarse model regions

## Status

Accepted as the central execution direction. Dense Qwen2 remains the first
correctness baseline; conditional execution is not claimed for that unchanged
checkpoint.

## Context

The useful idea is larger than a faster matrix kernel: a model is a graph of
coarse executable regions, and inference is an activation wave through a
formally eligible subgraph. Each request can independently choose what runs,
where it runs, and under which numerical profile.

Semantic labels such as `CSharp`, `.NET`, `Reasoning`, or `Math` are useful for
training, diagnostics, and residency policy. A label alone cannot authorize a
runtime skip. Arbitrarily skipping layers in an existing dense checkpoint
changes its function and is not a valid optimization.

## Decision

Synapse separates three immutable representations:

1. Model IR defines mathematical operations, state/effects, bounded control
   flow, entry points, and coarse regions.
2. Execution IR chooses layouts, lifetimes, kernels, memory spaces, and legal
   fusion groups for a numerical profile.
3. Deployment plans bind executable regions to devices/workers, budgets,
   transfers, session ownership, a plan hash, and an execution epoch.

A model region has explicit inputs, outputs, required tensors, state reads and
writes, and exactly one eligibility rule:

- always required;
- selected by a boolean value produced by the graph;
- selected by an immutable trained routing policy;
- optional only in an approximation profile tied to quality evidence.

Executable regions form a disjoint partition of nodes. Overlapping semantic
membership is expressed with annotations, while a shared producer lives in its
own region and may feed several downstream regions without being executed
twice. Unselected regions produce an explicit skipped outcome so fan-in waits
only for the branch semantics that were actually selected.

The activation scheduler works on fused regions or coarse subgraphs, never on
one Orleans grain per neuron, tensor tile, or graph node. It reserves bounded
memory before dispatch, tracks data/effect/state dependencies, and applies
backpressure to its bounded event queue.

Orleans owns distributed control-plane authority: sessions, epochs, leases,
placement, worker incarnations, fencing, and recovery. Large tensors, latent
payloads, and hot KV pages bypass Orleans serialization and use a direct
bounded worker transport. Local execution has the same region semantics but
requires neither Orleans nor a network.

Region-aware demand paging may keep only the active working set resident.
Eviction and prefetch decisions use immutable weight identities, expected next
regions, transfer cost, memory budgets, and active leases. A large on-disk
model with a small resident set is a research target, not a current product
claim.

Latent edges are ordinary typed graph values. Their shape, encoding,
numerical contract, source fingerprint, and destination compatibility are
verified exactly like other region boundaries.

## Quality and routing gates

Conditional models require a genuine trained or explicitly programmed route.
Evaluation compares full-graph execution with learned routing, random routing
at equal compute, and a static budget-matched route. It records route cost,
active weights/operations, memory residency, latency, C# task quality, general
quality, and recovery behavior.

Mixture-of-Depths, router tuning, early exit, JEPA-style latent prediction,
and speculative decoding are research references, not evidence that a Synapse
implementation is correct or faster. Exact papers, revisions, training
recipes, and measured results must be added to the attribution/evidence ledger
before a benchmark claim uses them.

## Consequences

- The Graph IR carries region and eligibility contracts now, before Orleans.
- The current dense Qwen2 graph marks every transformer region as required.
- Precision overlays, placement, and route selection remain independent axes.
- Route signatures are bounded compilation-cache keys; per-request unbounded
  compilation is forbidden.
- Orleans is introduced only with a runnable multi-process topology and direct
  data path; it is not used to hide local inference behind actor calls.
