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
- executable-region membership and explicit activation, provenance, and skip
  rules;
- exact region inputs, outputs, constant tensors, state reads, and state writes
  derived independently from member-node dependencies;
- operation-compatible typed attributes for normalization, RoPE, grouped
  causal attention, and top-k routing, including finite ranges and
  head/state shape agreement;
- an explicit scalar I32 `position` input for RoPE, state append, and causal
  attention rather than out-of-band decoder state.

All graph-owned collections are defensive read-only snapshots. Executable
regions are a disjoint partition of executable nodes; `Input`/`Output` nodes
are entry-point plumbing outside that partition. Each `Constant` is bound to a
unique `TensorId`. Semantic overlap is represented by annotations rather than
duplicate execution ownership.

Each `TensorId` also resolves to one `WeightDescriptor`: a package-relative
source file, overflow-checked offset/length, explicit physical encoding, and
logical shape. The verifier rejects missing, extra, duplicated, unsafe, or
shape/encoding-incompatible descriptors. Encoded-range content hashes remain
optional until the ZoneTree-backed F2 hash cache lands.

`ModelGraphFingerprint.Compute` writes every semantic Model IR field through a
versioned, explicitly little-endian, length-prefixed encoding and returns its
lower-case SHA-256 digest. Top-level declarations and set-like region fields
are ordered by their stable IDs; operation input order and entry-point argument
order remain significant. Unknown attribute or activation variants fail
instead of silently colliding with a known encoding.

## FlyBrain regions

Regions are coarse execution units with nodes, boundary values, required
weights, state effects, and a three-part `RegionActivation` contract. The
decision is `AlwaysActive`, a graph-produced boolean predicate, a slot of a
graph-produced `TopKRoute`, or an admission profile. Provenance records whether
skipping is structural, programmed, trained, or backed by approximation
evidence. Skip semantics declare whether outputs remain required, are absent,
or bypass to shape- and type-compatible region inputs. The verifier rejects
self-gating, decisions produced after their regions, out-of-range route slots,
non-causal sequence routing for step or
token decisions, absent outputs consumed by ordinary nodes, invalid bypasses,
and skippable state writers without position-hole-aware readers. Semantic
annotations such as `CSharp` and `Reasoning` do not affect control flow.

`Merge(Add)` and `Merge(SelectActive)` can consume absent values. A state slot
can allow skipped positions only when every reader masks those holes. The
canonical graph fingerprint includes all activation decisions, provenance,
bypass mappings, typed route parameters, and state-hole flags. These are
validated Model IR contracts; runtime branch scheduling and hole-aware kernels
are F1 work.

The first F1 scalar reference operators now compute row-major linear/bias,
causal grouped-query attention, RMSNorm, SiLU, element-wise Add/Multiply,
stable Softmax, and NeoX/interleaved RoPE with FP64 intermediates and FP32
outputs. The attention path reads only valid positions at or before the query position;
`AllKeysMasked`, active non-finite values, and FP32 overflow are typed failures
that leave the output unchanged. `NumericalPolicy` names the small-tensor FP32,
FP16, and BF16 tolerance contracts. A first `GraphReferenceInterpreter`
increment executes a verified, single-entry, fixed-shape, stateless,
always-active Model IR graph in node order. It accepts explicit FP32 input and
weight payloads, copies them before execution, and supports Linear, RMSNorm,
Add, Multiply, SiLU, and Softmax through scalar reference operators. Its
current numerical mode is FP32 storage/compute with declared FP64 accumulation;
other precision modes, conditional regions, state effects, symbolic shapes,
and unsupported ops fail before node execution. A one-million-element-per-
tensor safety bound keeps this correctness bridge small. Qwen execution from
IR, additional operations, and the region scheduler remain pending.

This represents the `what executes` axis. Execution IR will add kernels,
layouts, lifetimes, memory spaces, and precision. Deployment plans will bind
the resulting regions to local or remote devices and worker incarnations.

## Qwen2 integration and current boundary

The working Qwen2 loader now materializes its embedding region, every dense
transformer block, KV state/effect dependencies, and logits region as Model IR.
The real GGUF-backed graph must verify before scratch or KV allocation, and all
regions are `AlwaysActive` with structural provenance and `NotSkippable`. Its
GGUF epsilon, RoPE theta, head counts, head
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
and proves one model fingerprint. `RequiredWeightsResolveToSourceRanges`
checks every required tensor in the real Qwen graph; the missing-descriptor
regression proves the verifier fails closed. The shared Qwen smoke test verifies
its real 26-region graph before checking generation against dotLLM and
LLamaSharp. `RegionActivationTests` rejects eight invalid decision, skip, and
state combinations. `RegionActivationValidGraphTests` accepts valid feature
routing, tolerant merge, and hole-aware attention graphs, rejects an unaware
reader, and proves provenance changes the canonical fingerprint.
`ReferenceOperatorsTests` checks seeded and adversarial linear cases against
an independent FP64 calculation, causal future-value isolation, the typed
all-masked error, RMSNorm/SiLU values, stable large-logit softmax, both RoPE
layouts, and non-mutating shape errors.
`GraphReferenceInterpreterTests` executes real tiny Linear and RMSNorm→SiLU
graphs and rejects unsupported operators, conditional regions, FP32
accumulation, and short weight payloads.
