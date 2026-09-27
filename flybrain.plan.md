# FlyBrain execution plan

Date: 2026-09-27. Status: plan only. No step below is implemented unless it
is checked with evidence paths. Normative scope remains the development
specification, `docs/Architecture.md`, ADR-002, and accepted ADR-003.
Decision summary: `flybrain.brainstorm.md`.

## How Codex should use this plan

- Read `AGENTS.md`, ADR-002, ADR-003, `docs/Features/GraphExecution.md`, and
  this file before each step.
- Take one step at a time in order. Write the named failing test first, record
  red evidence, implement the smallest complete flow, run the related suites
  and the canonical gates, then check the step here with evidence paths and
  exit codes.
- Contract changes (`Synapse.Contracts`, schemas, ADRs) have one writer at a
  time. Do not mix a contract change with unrelated refactoring.
- Every speed or memory statement needs raw paired evidence from the benchmark
  runner. Until that exists, report measurements as development observations.
- Keep file, type, and function limits (400/250/60, nesting 4). Region kernels
  are the natural split point for the current large Qwen2 files.

## 1. Direction in one page

FlyBrain optimizes how much of the model runs and stays resident for a
request. It does this along four independent axes: what executes, when a
region is resident, where it runs, and with which precision. Each capability
enters only with a model that makes it legitimate:

| Level | Capability | Model that legitimately exercises it | Numerical mode | Gate |
|---|---|---|---|---|
| L0 | IR-driven region execution and tracing | Qwen2.5-0.5B Q8_0 (all regions required) | EquivalentNumerics | G01, G03 |
| L1 | Branch/merge/skip semantics, zero work when inactive | Synthetic `tiny-conditional` (spec §2.2) | ReferenceDeterministic | G01 |
| L2 | Residency and demand loading under a budget | Qwen2.5-0.5B with budget below its weights (capacity path) | EquivalentNumerics | G04 |
| L3 | Fewer target passes / layers per committed token, same output | External draft Qwen2.5-0.5B → Qwen2.5-Coder-1.5B (Apache); LayerSkip Llama 3.2 1B as local research only | Greedy target-equivalent | G12 |
| L4 | Exact expert routing and expert paging | OLMoE-1B-7B-0924-Instruct, official GGUF (Apache-2.0) | EquivalentNumerics | G04, G06 |
| L5 | Regions on several processes and nodes | Any L0–L4 model, split by layer ranges | EquivalentNumerics | G09, G14 |
| L6 | Precision per region | Q4/Q8 mixed profile from sensitivity | QualityBounded | G06, G07, G11 |
| L7 | Learned routing and domain regions | Router trained natively, domain experts | QualityBounded/Experimental | G07, G10 |

Facts that fix this order:

1. Skipping layers of an unchanged dense checkpoint changes its function. The
   first real "less of the model runs" results must come from checkpoints
   whose function already includes it: early-exit self-speculation (output
   equals the full model's greedy output) and MoE (unselected experts are
   outside the function).
2. In decode, the value crossing a layer boundary is one hidden vector per
   token, which is 3.5 KiB for Qwen2.5-0.5B in FP32. Layer-range placement
   ships that vector plus the position, and each region owner keeps its KV.
3. "C#" or "Math" regions are not separable in dense weights. Measure them on
   a real MoE first (route histograms per corpus). Train them only after native
   training exists.

## 2. Review findings on the current tree

Fix RV-1 through RV-4 before starting F1. The rest can ride along with the
step that touches the same code.

| ID | Severity | Where | Finding | Fix |
|---|---|---|---|---|
| RV-1 | Resolved 2026-09-28 | `AGENTS.md`, `docs/Architecture.md`, README | C# is the permanent portable/reference path and first implementation. Rust takes optimized kernels, allocator, hot-KV operations, or direct transfer only after paired profiling. The current C# and Rust KV implementations are reference and optimized-candidate roles, not competing authorities. |
| RV-2 | High | `Qwen2Model.cs` `Forward`/`ExecuteLayer` | Shadow IR: the graph is built and verified, but execution ignores it. The smoke test checks only the region count. | F1: execute through regions, then delete the hand-written layer loop. |
| RV-3 | Resolved 2026-09-28 | `GraphRegionVerifier`, `GraphRegionBoundaryVerifier`, Qwen2 graph builder | `Input`/`Output` are excluded from regions; every executable node is covered once. Inputs, outputs, state effects, and `TensorId`-bound constants are independently derived and compared with every descriptor. The real Qwen graph and explicit lying-descriptor regressions pass. |
| RV-4 | High | `GraphRegions.cs` | `TrainedRoute` has no decision value. There are no skip semantics for outputs or state. | F0.7 (ADR-003) |
| RV-5 | Medium | `Qwen2LayerGraphBuilder.AddAttention` | No operation attributes (epsilon, theta, heads, scale, mask) and no position value; `Rope([q])` and `CausalAttention([q])` are not executable from the IR. | F0.4 |
| RV-6 | Medium | `Qwen2LayerGraphBuilder` `Matrix(contextSize, …)` | The session context size is baked into Model IR state shapes, so graph identity depends on a session option. | F0.5 |
| RV-7 | Medium | `Qwen2GraphBuildContext.AddWeight` | `TensorId` is a counter with no source range, encoding, or content identity. Residency and overlays need one. | F0.6 |
| RV-8 | Medium | `GraphShapeVerifier.ValidateEmbedding` | Only a single token index `Fixed(1)` is accepted, so no prefill entry point with a bounded `Tokens[1..chunk]` dimension can exist. | With F1/F3 batched prefill |
| RV-9 | Resolved 2026-09-28 | `ModelPackageDownloader` | Streaming stops before exceeding declared size, `Content-Length` is checked, redirects are manual and capped at five, and only HTTPS source/Hugging Face storage hosts are trusted. Focused stream/trust tests plus a real redirected Hugging Face download pass. |
| RV-10 | Resolved 2026-09-28 | `ModelPackageCatalog`, downloader root check | Leading-dot IDs are rejected, and the package root plus every file must remain below the selected output root. Red/green regressions cover `.`, `..`, and `.hidden`. |
| RV-11 | Resolved 2026-09-28 | `.github/workflows/verify.yml` | SHA-pinned `actions/cache` v6 caches the verified model root by OS, RID, and catalog hash; fetch still re-hashes cache hits. |
| RV-12 | Low | `ReferenceSubjectsSmokeTests` | `Regions.Count == 26` breaks as soon as the layer is split into attention and MLP regions. | Assert the semantic properties instead: every region is always active/structural, full coverage, and a valid boundary. |
| RV-13 | Resolved 2026-09-28 | `README.md`, `docs/Development/Commands.md` | Examples fetch the smoke set and use the ignored `artifacts/models/...` path. |

## 3. Milestones

### F0. Make the IR honest and executable (completes TASK-GRF-001, ADR-003)

- [x] F0.1 ADR-003 approved as the execution-semantics direction on
  2026-09-28.
- [x] F0.2 `Input` and `Output` nodes are not region members; every other node
  is covered exactly once. The Qwen2 builder follows this boundary.
  Evidence: `EntryPlumbingNotRegionMember` and the real Qwen smoke test.
- [x] F0.3 Derive region inputs, outputs, state reads and writes, and required
  weights from member nodes, and compare them with the descriptor.
  Evidence: `RegionBoundaryDerivedAndCompared` rejects a lying descriptor
  with `RegionBoundaryMismatch`; the real Qwen2 graph passes.
- [ ] F0.4 Add typed operation attributes and an explicit `position` entry
  input consumed by `Rope`, `StateAppend`, and `CausalAttention`. The Qwen2
  builder reads epsilon, theta, and heads from GGUF metadata.
  Tests: `OperationAttributesRequired`, `PositionIsExplicitRegionInput`.
- [ ] F0.5 Use a bounded `Context` symbol in the state slot shapes, and add a
  canonical `ModelGraphFingerprint` (SHA-256 over a canonical encoding).
  Test: `ContextBoundInExecutionPlanOnly` (graphs built for context 256 and
  512 have the same fingerprint).
- [ ] F0.6 Add `WeightDescriptor` with a source file range and an encoding for
  each `TensorId`; region required weights resolve to descriptors. Content
  hashes can come in F2 (cached in ZoneTree).
  Test: `RequiredWeightsResolveToSourceRanges`.
- [ ] F0.7 Implement `RegionActivation` = decision + provenance + skip,
  `StateSlotDescriptor.PositionHolesAllowed`, and `MergeMode.SelectActive`,
  with the ADR-003 verifier rules. Migrate the existing tests.
  Tests: `TrainedRouteRequiresDecisionValue`, `DecisionProducerOutsideRegion`,
  `NonCausalDecodeRouteRejected`, `AbsentOutputRequiresTolerantConsumer`,
  `BypassShapeMustMatch`, `SkippableStateWriterRequiresHoleAwareReaders`,
  `AlwaysActiveMustBeNotSkippable`.
- [ ] F0.8 Update `docs/Features/GraphExecution.md`, mark ADR-003 accepted,
  and record evidence for TASK-GRF-001.

Exit: GRF-001 AC 1–3 plus the new tests pass, and the Qwen smoke test still
produces token `12095`. Not claimed: execution from the IR.

### F1. The IR is what runs (TASK-GRF-002, TASK-GRF-003, tiny part of TASK-CND-001)

- [ ] F1.1 A reference interpreter: scalar FP32 per operation with optional
  FP64 accumulation, covering every operation that the Qwen2 and tiny
  fixtures use. Tests: `OperatorsMatchFp64Oracle`,
  `CausalMaskPreventsFutureLeak`, `AllMaskedRowDefined`.
- [ ] F1.2 A region scheduler. Precompute the topological region order; keep
  indegree counters and a bounded ready queue for DAG fan-out. Outcomes are
  `Executed`, `Skipped`, `Bypassed`, and `Absent`. Cancellation is checked
  between regions, and boundary buffers are preallocated from lifetimes (no
  allocation per step).
- [ ] F1.3 A kernel registry bound by `RegionPattern`. Split Qwen2 into
  `Embedding`, `Attention(layer)`, `Mlp(layer)`, and `Logits` region kernels,
  giving two regions per layer. Kernels take their parameters from IR
  attributes. Tests: `RegionPatternMismatchRejected`,
  `OperationAttributesDriveKernels`.
- [ ] F1.4 `Qwen2Model.Generate` runs through the scheduler, and
  `ExecuteLayer` is deleted. Before deletion, record paired decode-step
  timings of the old and new paths on the same machine as development evidence
  (target: scheduler overhead is small; the number is reported, not claimed).
  Tests: `QwenViaRegionsMatchesBaselines` (token `12095`, and the same N greedy
  tokens as before), `FusedRegionsMatchReferenceInterpreter` (per-region
  hidden-state diff for the first three positions within the declared
  tolerance).
- [ ] F1.5 `ActivationTrace` (bounded, opt-in) and
  `synapse generate --trace-regions <file.jsonl>`. See §5 for the schema.
- [ ] F1.6 A deterministic `tiny-conditional` fixture generated by repo code
  (shared stem, four experts, top-1 router, two heads, known outputs).
  Tests: `InactiveBranchDoesNoWork` (kernel and bytes-read counters),
  `MergeHandlesSkippedInput`, `LoopBudgetAndCancelWork`,
  `OnlySelectedExpertExecutes`, `RouteTieStable`, `CapacityOverflowDefined`.

Exit: the Qwen2 path has no code outside the region kernels, and all the
tests above pass. Not claimed: speedup, or any conditional real model.

### F2. Reservation ledger and region residency (TASK-MEM-001, TASK-MEM-003 subset)

- [ ] F2.1 A reservation ledger with the spec §7.1 categories. Admission fails
  with `InsufficientMemory` and the calculated requirement.
- [ ] F2.2 Decide the residency mechanism in ADR-005 (open decision D5). The
  recommendation is explicit, aligned, owned buffers filled with
  `RandomAccess.ReadAsync` for the paged path, which gives a hard budget and
  counted bytes loaded. Keep the read-only mmap for the fully resident path.
- [ ] F2.3 A weight residency manager. A chunk is one tensor. States follow
  spec §7.2; a pin is held while a region executes; eviction is a
  cost-aware LRU; prefetch looks a bounded K regions ahead in plan order.
  Per-tensor content hashes are cached in ZoneTree.
- [ ] F2.4 Capacity mode: Qwen2.5-0.5B with a weight budget of about 40% of
  its weights produces the same greedy tokens. The trace shows loads per step.
  Tests: `InFlightChunkNotEvicted`, `InterruptedLoadRetryable`,
  `PrefetchStaysBounded`, `CapacityModePreservesTokens`,
  `BudgetBelowLargestRegionRejected`.

Exit: the budget is enforced, and bytes loaded per token are reported. Not
claimed: fast paging, or "large model in small RAM" as a speed result.

### F3. Llama family and tokenizer (TASK-PKG-003, TASK-PKG-004, TASK-GEN-003)

Prerequisite for F4 (LayerSkip checkpoints are Llama-architecture) and for
real prompts.

- [ ] F3.1 A Llama graph builder on the same region contract. Kernels bind by
  pattern, so Llama attention without bias is a distinct pattern or an
  attribute.
- [ ] F3.2 SmolLM2-135M from safetensors BF16 (reference BF16 path first).
  Test: `SmolLm2MatchesBaseline` (greedy tokens against LLamaSharp and dotLLM
  on a pinned prompt).
- [ ] F3.3 A repo-owned BPE tokenizer and explicit chat templates for Qwen2 and
  Llama 3. Tests: `TokenizerGoldenRoundTrip`, `ChatTemplateGolden`.

### F4. Fewer target passes per committed token, same output (TASK-SES-002, TASK-SPC-001, then TASK-SPC-004 moved earlier)

The first real FlyBrain result: less target work per committed token, with
unchanged output.

License constraint: the published LayerSkip checkpoints use the FAIR
Noncommercial Research License and manual gated access. They are not CI
fixtures, not redistributed, and not the product path. The Apache path is an
external draft from the same tokenizer family.

- [ ] F4.1 A session working branch: `BeginBranch`, `CommitPrefix(k)`, and
  `RollbackTail`, with KV holes allowed only in the branch (ADR-003 §2).
  Tests: `RejectedDraftLeavesNoCommittedState` (reject at first, middle, and
  last), `CommitRequiresCompleteState`.
- [ ] F4.2 Greedy verification with an external draft (SPC-001): Qwen2.5-0.5B
  drafts for Qwen2.5-Coder-1.5B (spec profile `qwen-coder-local`). Token-ID
  identity is verified by tokenizer hash, not assumed. The draft and target are
  two region graphs in one plan, which is also the first multi-model activation
  wave. Test: `ExternalDraftGreedyEquivalent`.
- [ ] F4.3 Self-speculation (SPC-004) as a local research run on
  `facebook/layerskip-llama3.2-1B` (`LlamaForCausalLM`; the owner accepts the
  gate and license). A `draft(E)` entry point runs regions L0..L(E-1) plus the
  shared LM head. `verify` runs L(E)..L(n-1) for the draft block and reuses the
  early-layer KV and the exit-layer query (the paper's KVQ cache). E comes from
  the graph and profile, never a constant. Whether the final norm is applied at
  the exit must be read from the checkpoint's reference code before
  implementation. Tests: `SelfSpeculationGreedyEquivalent` (N prompts × M
  tokens identical to non-speculative greedy), `DraftKvReusedNotRecomputed`
  (early-layer region executions per committed token from the trace).
- [ ] F4.4 A control: the same exit layer on Apache Llama/Qwen checkpoints
  without early-exit training. It shows why training matters, and its
  acceptance rate is reported.

Exit: G12 on the external-draft pair through the benchmark runner
(TASK-BMK-001 and TASK-BMK-005 are prerequisites for the claim). Report the
acceptance rate, target and draft region executions per committed token, and
committed tokens per second. Self-speculation results stay labeled as
noncommercial research evidence.

### F5. Real MoE: exact expert routing and expert paging (TASK-CND-001 real level C1, TASK-CND-004, TASK-MEM-003)

- [ ] F5.1 Pin `allenai/OLMoE-1B-7B-0924-Instruct-GGUF`: Apache-2.0, not
  gated, `OlmoeForCausalLM`, 16 layers, 64 experts with top-8 per layer, 6.9B
  total and 1.3B active parameters, `norm_topk_prob: false`. Choose one
  encoding that LLamaSharp can run as the baseline, and verify that LLamaSharp
  0.27.0 loads it before any implementation. The alternative is
  `ibm-granite/granite-3.1-1b-a400m-instruct` (Apache-2.0,
  `GraniteMoeForCausalLM`, 24 layers, 32 experts, top-8), imported from
  safetensors because IBM publishes no first-party GGUF.
- [ ] F5.2 Graph per layer: an attention region, a router region
  (`TopKRoute` with k, tie policy, `normalize_top_k`, and capacity
  attributes), 64 expert regions with `RouteSlotDecision` + `Structural` +
  `OutputsAbsent`, and `Merge(GatedSum)`. That is more than 1,000 regions per
  step, so the scheduler, trace, and route-signature cache must stay bounded.
  Decode uses `Step` scope. Prefill uses `TokenInBatch` with gather/scatter per
  expert (region-level batching). Tests: `MoeGreedyMatchesBaseline`,
  `OnlySelectedExpertsReadBytes` (exactly 8 expert regions per layer read
  weights).
- [ ] F5.3 An expert cache under a budget smaller than all experts: an LRU with
  k experts per layer plus speculative prefetch that applies the next layer's
  gate to the current hidden state (Eliseev & Mazur 2023). A prefetch miss
  waits and never changes outputs. Tests: `SessionUnionMemoryMeasured`,
  `PrefetchPredictionMissSafe`, `SparseBenefitSeparatesOverheads`.
- [ ] F5.4 Experiment R-DOM-1, registered before running (spec Part K
  contract): route histograms per corpus (C#, other code, English prose, math)
  from traces. The OLMoE paper already reports strong specialization for some
  domains (arXiv, GitHub) and a near-uniform spread for C4. R-DOM-1 asks the
  C#-specific question on our own traces. The result is a report, not a product
  claim, and it is the first data-driven answer to whether a "CSharp region"
  exists.
- [ ] F5.5 Later: `Qwen/Qwen3-30B-A3B` (Apache-2.0, 48 layers, 128 experts,
  top-8, 30.5B total and 3.3B active parameters) on the Mac under a fixed
  budget. It is a capacity-only result with measured tokens per second.

### F6. Regions across processes, then Orleans (TASK-TRN-001..003, TASK-CLS-001..004)

- [ ] F6.1 A local two-process pipeline. Worker A owns layer regions [0, k),
  and worker B owns [k, n) plus the logits; each owns its KV. The packet per
  decode step is the hidden vector plus `(session, epoch, step, position,
  plan hash)` in data frame v0 (spec §8.3). Sampling happens at the final
  stage. Tests: `TwoProcessTokensEqualSingleProcess`,
  `StaleEpochFrameRejected`, `WrongPlanHashRejected`.
- [ ] F6.2 Orleans after F6.1 works, following §4. Test:
  `NoGrainCallsInSteadyStateDecode` (trace topology: zero grain calls per
  100 steady-state tokens).
- [ ] F6.3 Two-node evidence on H-DUAL-LAN. Report capacity and latency
  separately.

### F7. Precision per region (TASK-QNT-001..004, then TASK-PRE-001..004)

- [ ] F7.1 Q4 `syn.q4.symmetric.g64.v1` encoding and kernel (reference path
  first).
- [ ] F7.2 The Execution IR chooses the encoding per region from a
  session-scoped `PrecisionProfile` (`ProfileDecision`). The package stores the
  needed encodings, and the region pattern includes the encoding.
- [ ] F7.3 A static mixed profile from sensitivity (spec §6.2) compared with
  uniform Q4 and Q8 at equal bytes. This needs the sealed quality suite
  (TASK-BMK-003).
- [ ] F7.4 Domain residual packs (the C# overlay) only as research R02, after
  F7.3.

### F8. Learned routing and domain regions (research: TASK-ADP-001, TASK-CND-002/003, R01)

- [ ] F8.1 Native router-only training (ADP-001) in the Router-Tuning/MindSkip
  style. The trained part is one d→1 router per layer with the backbone frozen.
  The paper evaluates Qwen-2.5-7B/14B among others; its first form skips
  attention only, and later versions also cover MLP, whole blocks, and MoE
  experts. Training still needs a native backward pass through the frozen
  forward operations, so ADP-001 must cover those gradients (finite-difference
  oracle). Skipped attention leaves KV holes (ADR-003 §2), matching MoD, where
  skipped tokens contribute no keys or values at that block. Provenance is
  `TrainedPolicy`, and the controls are full, random, static, and learned.
- [ ] F8.2 Domain experts merged into an MoE in the Branch-Train-MiX style:
  separately trained math, code, and wiki experts become FFN experts with a
  learned router. Start only once native fine-tuning exists. This is the
  evidence-backed path to a real "CSharp region".

## 4. Orleans mapping of the region idea

| Grain | Key | Owns | Called when |
|---|---|---|---|
| `SessionGrain` | SessionId | Owner worker, epoch, durability mode, plan hash | Create, cancel, recover, replan at a safe point |
| `RegionPlacementGrain` | (model fingerprint, placement ID) | Region range → worker incarnation, leases, residency summary | Deployment, failure, replan |
| `WorkerRegistryGrain` | WorkerId | Capabilities, incarnation, health | Worker start/stop/health |

The activation wave never goes through a grain. Workers forward packets
directly using the plan fixed for the session epoch. A stale epoch or wrong
plan hash is rejected at the receiver. Local mode uses the same scheduler and
placement types in process, with no Orleans assembly loaded (INV-006).

## 5. Activation trace and metrics

One bounded, opt-in JSONL event per region per step:

```json
{"session":"…","step":12,"position":17,"region":34,"kind":"Mlp","layer":16,
 "outcome":"Executed","provenance":"Structural","durationNs":0,
 "weightBytesTouched":0,"weightBytesLoaded":0,"device":"cpu","worker":"local",
 "precision":"q8_0"}
```

Run-level aggregates: regions executed and skipped per token; bytes touched,
resident (peak), loaded, and transferred reported separately (spec §1.3);
route and expert histograms keyed by region ID only (bounded cardinality);
expert cache hit rate; draft acceptance rate. A later artifact can render the
wave as a token × region heatmap from this file.

## 6. Do not

- Do not skip regions of a dense checkpoint because of a label, a keyword, or
  activation magnitude.
- Do not keep a second executor beside the region scheduler after F1.4.
- Do not create grains per token, region activation, or neuron, and do not
  pass tensors through Orleans.
- Do not repeat paper speedups as Synapse targets or results.
- Do not mark a step done without red and green evidence and exit codes.

## 7. Decisions and remaining owner choices

- D1 resolved 2026-09-28: C# is the first and permanent portable/reference
  execution path; Rust owns a proven optimized boundary, including hot KV or
  transfer, only when profiling justifies it.
- D2 resolved 2026-09-28: ADR-003 is accepted as the implementation direction.
- D3: Move TASK-SPC-001 (and TASK-SPC-004 as local research) ahead of
  TASK-SPC-003 and the cluster milestones, because it is the first exact
  FlyBrain win and it exercises the working-branch KV that everything else
  needs.
- D4: Accept or reject local research use of the gated FAIR-noncommercial
  LayerSkip checkpoint (F4.3). Confirm OLMoE-1B-7B-0924-Instruct as the F5
  fixture.
- D5: Choose the residency mechanism in ADR-005 (owned buffers vs.
  mmap+madvise).

## 8. Research references (checked 2026-09-27)

These are design references, not Synapse evidence. Their published speedups
are not Synapse targets.

- LayerSkip, arXiv 2404.16710. Checkpoints are in the
  `facebook/layerskip-*` collection (FAIR Noncommercial Research License,
  gated), with a shared LM head across exits and KVQ cache reuse.
  https://arxiv.org/abs/2404.16710,
  https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a
- Mixture-of-Depths, arXiv 2404.02258. Top-k over the sequence is non-causal;
  sampling uses an auxiliary loss or predictor; skipped tokens contribute no
  K/V at that block. https://arxiv.org/abs/2404.02258
- Router-Tuning/MindSkip, arXiv 2410.13184. Router-only training with a frozen
  backbone. https://arxiv.org/abs/2410.13184,
  https://github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths
- OLMoE, arXiv 2409.02060, including its domain-specialization analysis.
  https://huggingface.co/allenai/OLMoE-1B-7B-0924,
  https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF
- Granite 3.1 MoE:
  https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct
- Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
- MoE offloading with an LRU expert cache and speculative expert loading,
  arXiv 2312.17238. https://arxiv.org/abs/2312.17238
- Branch-Train-MiX, arXiv 2403.07816; Branch-Train-Merge, arXiv 2208.03306;
  DEMix, arXiv 2108.05036.
