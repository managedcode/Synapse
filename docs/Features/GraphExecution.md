# Graph execution

## Implemented Model IR core

The portable C# contract now has distinct strong identities for nodes, values,
tensors, regions, entry points, state slots, effect tokens, graph versions,
and operation-set versions. Tensor values carry bounded shapes plus separate
storage, compute, and accumulator types.

The verifier executes before weight or execution-memory allocation and checks:

- bounded positive dimensions, rank limits, and element-count overflow;
- unique IDs and single-producer SSA values/effects;
- declared producer-before-consumer order and unstructured cycles;
- initial linear, element-wise, unary, reshape, and merge shape rules;
- structured loop bounds and carried-state arity;
- entry-point completeness;
- unknown state references, unwritten reads, and unordered writers;
- executable-region membership and formal eligibility rules;
- exact region inputs, outputs, constant tensors, state reads, and state writes
  derived independently from member-node dependencies;
- operation-compatible typed attributes for normalization, RoPE, and grouped
  causal attention, including finite ranges and head/state shape agreement;
- an explicit scalar I32 `position` input for RoPE, state append, and causal
  attention rather than out-of-band decoder state.

All graph-owned collections are defensive read-only snapshots. Executable
regions are a disjoint partition of executable nodes; `Input`/`Output` nodes
are entry-point plumbing outside that partition. Each `Constant` is bound to a
unique `TensorId`. Semantic overlap is represented by annotations rather than
duplicate execution ownership.

`ModelGraphFingerprint.Compute` writes every semantic Model IR field through a
versioned, explicitly little-endian, length-prefixed encoding and returns its
lower-case SHA-256 digest. Top-level declarations and set-like region fields
are ordered by their stable IDs; operation input order and entry-point argument
order remain significant. Unknown attribute or eligibility variants fail
instead of silently colliding with a known encoding.

## FlyBrain regions

Regions are coarse execution units with nodes, boundary values, required
weights, state effects, and one explicit eligibility rule. A region is either
always required, controlled by a scalar boolean graph value, selected by an
immutable trained-policy hash, or tied to immutable approximation evidence.
Semantic annotations such as `CSharp` and `Reasoning` do not affect control
flow by themselves.

This represents the `what executes` axis. Execution IR will add kernels,
layouts, lifetimes, memory spaces, and precision. Deployment plans will bind
the resulting regions to local or remote devices and worker incarnations.

## Qwen2 integration and current boundary

The working Qwen2 loader now materializes its embedding region, every dense
transformer block, KV state/effect dependencies, and logits region as Model IR.
The real GGUF-backed graph must verify before scratch or KV allocation, and all
regions are `AlwaysRequired`. Its GGUF epsilon, RoPE theta, head counts, head
dimension, attention scale/mask, and current decode position are explicit in
the graph. KV state capacity is `Context[1..model_max_context]`; the smaller
session allocation remains outside Model IR, so loading the same checkpoint at
256 and 512 tokens produces the same fingerprint. The optimized executor still
invokes its managed C# kernels directly; a general DAG executor and Execution
IR are the next boundary. Conditional routing, demand paging, latent region
edges, and Orleans placement are not claimed until the corresponding trained/
evaluated graph and execution/deployment plans exist.

## Acceptance mapping

`TypedGraphIrTests` covers a valid linear region, shape mismatch with NodeId,
multi-node and self cycles, unordered state writers, a non-boolean region
predicate, caller-owned collection mutation, entry-plumbing exclusion, lying
region boundaries, missing constant tensor identity, and overlapping execution
ownership. `OperationAttributesRequired` rejects hidden operator parameters;
`PositionIsExplicitRegionInput` checks the real model's position wiring and
typed GGUF-derived attributes. `CanonicalFingerprintStableAndSensitive` checks
canonical set ordering and semantic sensitivity;
`ContextBoundInExecutionPlanOnly` loads the real GGUF at two session capacities
and proves one model fingerprint. The shared Qwen smoke test verifies its real
26-region graph before checking generation against dotLLM and LLamaSharp.
