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
- executable-region membership and formal eligibility rules.

All graph-owned collections are defensive read-only snapshots. Executable
regions are a disjoint node partition; semantic overlap is represented by
annotations rather than duplicate execution ownership.

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
regions are `AlwaysRequired`. The optimized executor still invokes its managed
C# kernels directly; a general DAG executor and Execution IR are the next
boundary. Conditional routing, demand paging, latent region edges, and Orleans
placement are not claimed until the corresponding trained/evaluated graph and
execution/deployment plans exist.

## Acceptance mapping

`TypedGraphIrTests` covers a valid linear region, shape mismatch with NodeId,
multi-node and self cycles, unordered state writers, a non-boolean region
predicate, caller-owned collection mutation, and overlapping execution
ownership. The shared Qwen smoke test also verifies its real 26-region graph
before checking generation against dotLLM and LLamaSharp.
