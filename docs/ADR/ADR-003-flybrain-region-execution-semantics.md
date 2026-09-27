# ADR-003: FlyBrain region execution semantics

## Status

Accepted on 2026-09-28 as the implementation direction after the
planning/review pass in `flybrain.plan.md`. The public `Synapse.Contracts`
changes remain unimplemented until their F0 red/green behavior tests land. It
extends ADR-002 and does not replace it.

## Context

ADR-002 made coarse regions the unit of the activation wave. The current
contract (`GraphRegions.cs`) can say *that* a region is optional, but not
*how* the runtime decides, *what* its outputs are when it does not run, or
*what* happens to state it would have written. Without those rules an
executor cannot skip anything safely, so every FlyBrain capability (routing,
early exit, expert paging, placement, precision per region) would have to
invent private semantics.

The concrete gaps in the current code:

1. `TrainedRouteEligibility(PolicyHash)` names a policy but no graph value that
   carries its decision. The runtime cannot evaluate it.
2. There are no skip semantics. A skipped residual block should pass its input
   through. A skipped MoE expert contributes nothing to the merge. A skipped
   attention block does not append KV. None of these can be expressed.
3. Region boundaries (`Inputs`, `Outputs`, `StateReads`, `StateWrites`,
   `RequiredWeights`) are declared by the builder and never checked against
   the member nodes, so a descriptor can be wrong without detection.
4. Operations carry no attributes: epsilon, RoPE theta/layout, head counts,
   attention scale, and mask. Positions are also implicit. So the IR cannot be
   executed by itself; the executor needs out-of-band parameters.
5. `TensorId` is a counter. It is not linked to a source file range, encoding,
   or content hash, so residency, paging, and precision overlays have no
   stable weight identity.
6. KV state shapes bake the session context size into the Model IR, which
   mixes model identity with session configuration.

## Decision

### 1. Split eligibility into decision, provenance, and skip semantics

```csharp
public sealed record RegionActivation(
    ActivationDecision Decision,
    EligibilityProvenance Provenance,
    SkipSemantics Skip);

// How the runtime decides, evaluated by the scheduler.
public abstract record ActivationDecision;
public sealed record AlwaysActive : ActivationDecision;
public sealed record PredicateDecision(ValueId Predicate, RouteScope Scope) : ActivationDecision;
public sealed record RouteSlotDecision(ValueId Route, int Slot, RouteScope Scope) : ActivationDecision;
public sealed record ProfileDecision(string ProfileKey) : ActivationDecision;

public enum RouteScope { Session, Step, TokenInBatch }

// Why skipping is legitimate. Checked at admission, never at decode time.
public abstract record EligibilityProvenance;
public sealed record StructuralProvenance : EligibilityProvenance;   // the model's own function (dense, MoE router)
public sealed record ProgrammedProvenance : EligibilityProvenance;   // graph-author predicate (C0)
public sealed record TrainedPolicyProvenance(ContentHash PolicyHash) : EligibilityProvenance;
public sealed record ApproximateProvenance(ContentHash EvaluationHash) : EligibilityProvenance;

// What the region's outputs are when it does not execute.
public abstract record SkipSemantics;
public sealed record NotSkippable : SkipSemantics;
public sealed record OutputsAbsent : SkipSemantics;
public sealed record BypassOutputs(IReadOnlyList<ValueBypass> Map) : SkipSemantics;
public readonly record struct ValueBypass(ValueId Output, ValueId Input);
```

Mapping from the current contract:

| Current | New |
|---|---|
| `AlwaysRequired` | `AlwaysActive` + `Structural` + `NotSkippable` |
| `GraphPredicate(v)` | `PredicateDecision(v, Step)` + `Programmed` + explicit skip |
| `TrainedRoute(h)` | needs a decision value: `Predicate`/`RouteSlot` + `TrainedPolicy(h)` |
| `ApproximateProfile(h)` | `ProfileDecision(key)` + `Approximate(h)` + explicit skip |

The effective numerical mode of a plan is derived from the provenance of its
possibly skipped regions. `Structural` and `Programmed` keep the model's
defined function. Skipping an unselected expert of a real MoE checkpoint is
therefore exact, not an approximation. `TrainedPolicy` and `Approximate` make
the plan `QualityBounded`, or `Experimental` without evidence. A session must
opt into that mode, or admission rejects the plan. This is how
`AC-CND-003-1 DenseModelCannotBeArbitrarilySkipped` is enforced.

### 2. State written by a skippable region

This matches Mixture-of-Depths (arXiv 2404.02258): a token that skips a block
contributes no keys or values at that block, so later tokens cannot attend to
it there.

`StateSlotDescriptor` gains `PositionHolesAllowed`. A skippable region may
write only slots that allow holes, and every reader of such a slot must be
hole-aware (attention with a per-position validity mask). Otherwise the
verifier rejects the graph. If a method needs state even when the block is
skipped, the graph splits that work into a separate, always-active region.
There is no hidden "compute KV anyway" flag.

Speculative drafts may leave holes only in the session's working branch.
`CommitPrefix` asserts that every slot without `PositionHolesAllowed` is
complete for the committed positions.

### 3. Boundaries are derived, then compared

`Input` and `Output` nodes are entry-point plumbing and are not region
members. Every other node belongs to exactly one region. For each region the
verifier derives:

- inputs: values consumed by member nodes but produced outside the region;
- outputs: values produced inside and consumed outside, or returned by an entry
  point;
- state reads and writes: the union over member nodes;
- required weights: the `TensorId`s of member `Constant` nodes.

A declared descriptor that differs is rejected with `RegionBoundaryMismatch`.

### 4. Decision-value rules

- The decision value is produced outside the region, and the region cannot
  reach its producer. This forbids self-gating and cycles.
- `PredicateDecision` needs a Bool value shaped `[1]` for `Step` or
  `[tokens]` for `TokenInBatch`. `RouteSlotDecision` needs a `TopKRoute`
  output and a slot within its declared range.
- A `Step` or `TokenInBatch` decision used by an autoregressive decode entry
  point must not depend on a `TopKRoute` over the sequence axis, because that
  is non-causal. Such routing must use a causal predictor.
- `AlwaysActive` requires `NotSkippable`, and every other decision requires a
  skip semantics other than `NotSkippable`.
- For `OutputsAbsent`, every outside consumer must tolerate absence:
  `Merge(SelectActive)`, or `Merge(Add)` over active inputs. `MergeMode` gains
  `SelectActive`, meaning exactly one active input passes through (early-exit
  heads).
- For `BypassOutputs`, each pair has the same shape and numeric type, and the
  input is a derived region input.

### 5. Executable IR: attributes, positions, bounded context, weight identity

- `GraphNode` gains typed per-operation attributes: RMSNorm epsilon; RoPE
  theta, layout, and head dimension; attention query heads, KV heads, scale,
  and mask kind; and `TopKRoute` k, axis, tie policy, and capacity. Kernels
  read these values from the IR, never from model-specific fields.
- Decode entry points take an explicit `position` input. `Rope`,
  `StateAppend`, and `CausalAttention` consume it, and it crosses region
  boundaries like any other value. The activation packet carries it
  explicitly.
- State slot shapes use a bounded symbol such as
  `Context[1..model_max_context]`. The session's context size is bound in the
  Execution IR, so the Model IR hash does not depend on session options.
- `ModelGraph` gains `Weights: IReadOnlyList<WeightDescriptor>`, where each
  entry is `(TensorId, SourceRange(file, offset, length), Encoding,
  LogicalShape, ContentHash?)`. Per-tensor content hashes are computed once
  per package and cached in ZoneTree under
  `(package sha256, offset, length)`.

### 6. Execution binding: no shadow executor

A fused region kernel is registered against a canonical `RegionPattern`
fingerprint. The fingerprint covers ordered operation kinds, internal edges,
shapes, and numeric types; attributes are parameters. Lowering binds each
region to a kernel or fails with `UnsupportedRegionPattern`. The
op-by-op reference interpreter is always available as the oracle and as the
reference backend. The hand-written Qwen2 layer loop is removed once the
region kernels replace it.

## Alternatives rejected

- Keep one `ExecutionEligibility` union. It forces provenance and runtime
  decisions into one type and leaves `TrainedRoute` unexecutable.
- Implicit residual bypass for any skipped block. It is wrong for MoE
  experts, heads, and non-residual regions, and it hides the model-function
  change.
- A "compute KV even when skipped" flag. That is a region split, and it
  should be visible in the graph and in the cost model.
- A Model IR that only exists for verification while a separate executor
  runs. The graph and the executor drift silently, and nothing in FlyBrain can
  attach to it.

## Consequences

- `Synapse.Contracts` changes, and both current region tests and the Qwen2
  builder need updates in the same change.
- The first executor that consumes this contract is `TASK-GRF-003`. The first
  real non-dense uses are early-exit self-speculation and real MoE expert
  routing (`flybrain.plan.md` F4/F5).
- A dense checkpoint keeps every region `AlwaysActive`/`Structural`. This ADR
  adds no route that a dense model could use without evidence.

## Verification

New acceptance scenarios are tracked in `flybrain.plan.md` under F0/F1:
`RegionBoundaryDerivedAndCompared`, `EntryPlumbingNotRegionMember`,
`TrainedRouteRequiresDecisionValue`, `DecisionProducerOutsideRegion`,
`NonCausalDecodeRouteRejected`, `AbsentOutputRequiresTolerantConsumer`,
`BypassShapeMustMatch`, `SkippableStateWriterRequiresHoleAwareReaders`,
`OperationAttributesDriveKernels`, `ContextBoundInExecutionPlanOnly`,
`RegionPatternMismatchRejected`, and `SkippedProvenanceSetsNumericalMode`.
