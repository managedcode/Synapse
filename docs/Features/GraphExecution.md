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

## Current boundary

The working Qwen2 executor has not yet been lowered through this IR. Its dense
checkpoint therefore remains fully required. Conditional routing, demand
paging, latent region edges, and Orleans placement are not claimed until the
corresponding trained/evaluated graph and execution/deployment plans exist.

## Acceptance mapping

`TypedGraphIrTests` covers a valid linear region, shape mismatch with NodeId,
an illegal cycle, unordered state writers, a non-boolean region predicate,
caller-owned collection mutation, and overlapping execution ownership. These
tests exercise the real verifier and run in the shared .NET gate.
