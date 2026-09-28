# Elastic inference plan

Date: 2026-09-28. Scope: concurrent requests, loading only the needed model
parts, on-the-fly importance-aware precision with automatic restore, a
heterogeneous Orleans cluster, and import from several model formats into the
Synapse graph format. Related plans: `flybrain.plan.md` (regions, residency,
placement) and `kv-performance.plan.md` (hot KV). Normative scope remains the
development specification.

## 0. Owner direction (2026-09-28)

- Serve many requests in parallel.
- Load only the parts of the model that a request needs.
- Quantize on the fly. Weights that matter for a correct result keep full
  precision; weights that do not matter may run approximately (as low as
  1-bit/ternary was the example). Precision must return automatically when a
  part becomes important.
- Orleans distributes work across a cluster of machines that differ in speed
  and architecture, so one model can be split across them.
- Convert models from different formats into the Synapse graph-based format.

## 1. Implemented building blocks (2026-09-28)

| Component | Files | Tests | Status |
|---|---|---|---|
| Symmetric codec family `syn.q{bits}.symmetric.g{group}.v1` for any width from 2 to 8 bits and any group size that is a multiple of 8 up to 1024: FP16 scale, LSB-first packed codes, linear without a dequantized tensor | `src/Synapse.Runtime/Features/Quantization/SymmetricGroupCodec.cs` | `SymmetricGroupCodecTests` (8): 3-bit golden bytes, error ≤ scale/2 for every width 2–8 and groups 8–128, strictly lower distortion with more bits, linear parity for every width, reserved code, invalid parameters, codec resolution by ID | Green |
| `syn.q4.symmetric.g64.v1` (spec §6.1) as the 4-bit/64 member of that family, byte-identical to the spec layout | `SynQ4BlockCodec.cs` | `QuantizedBlockCodecTests` (9): golden bytes, zero group, padding, error ≤ scale/2, invalid scale, reserved code, non-finite and overflow, exact size, linear parity | Green |
| `syn.ternary.absmean.g64.v1` codec (BitNet b1.58 absmean rule per group, 2.25 stored bits/weight), addition-only linear | `TernaryBlockCodec.cs` | `TernaryBlockCodecTests` (5) | Green |
| Codec registry by encoding ID (`WeightCodecs.TryGet`), and per-tensor options measured on the fly for any set of codecs (`PrecisionCandidates.Measure`) | `WeightCodecs.cs`, `PrecisionCandidates.cs` | `PrecisionCandidatesTests` (3): size order across 8/6/4/3/2-bit, equal-size options keep lower distortion, encodings not smaller than the source are dropped | Green |
| Source-format decoders to the FP32 reference: F32/F64/F16/BF16, FP8 E4M3FN and E5M2, GGUF Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL/MXFP4, Q2_K–Q6_K, Q8_K, IQ4_XS, TQ1_0, TQ2_0; typed rejection of unverified types | `src/Synapse.Runtime/Features/ModelPackages/SourceFormats/*` | `SourceTensorDecoderTests` (10), plus `GgmlReferenceDecoderTests` (3), which diff every implemented quant against ggml's own `dequantize_row_*` (16 random blocks per type; multiplicative types bit-exact, affine types within 2⁻²⁰ of the largest value) | Green |
| Activation-aware importance (spec §6.2 output distortion) | `WeightSensitivity.cs` | `WeightSensitivityTests` (5), including `DistortionIsActivationAware` | Green |
| Budgeted precision per tensor with device kernel support, pinning, an approximation opt-in, profile identity hash, and promotion diff | `PrecisionBudgetSelector.cs`, `PrecisionProfiles.cs` | `PrecisionBudgetSelectorTests` (11), including `LargerBudgetOnlyPromotes` and `PromotionRestoresMostImportantDemotedTensorsFirst` | Green |
| Heterogeneous layer-range placement: up to four devices, every device order, a DP split, per-stage KV reservation and importance-aware precision, latency or throughput objective | `src/Synapse.Runtime/Features/ExecutionPlanning/*` | `HeterogeneousPlacementPlannerTests` (7) | Green |

Evidence:

- `dotnet build Synapse.slnx --configuration Release` succeeded with zero
  warnings.
- The new suites passed 61/61: 41 quantization, 7 planning, and 13
  source-decoder tests.
- The full suite passed 128 of 132. All 4 failures are reference-baseline smoke
  tests that need `SYNAPSE_DOTLLM_EXECUTABLE` (1 test) or
  `SYNAPSE_LLAMACPP_EXECUTABLE` (3 tests). Those variables are not configured in
  this shell, and the tests are unrelated to these files.
- A deliberate sign mutation in the Q4_K decoder made the ggml differential test
  fail (maximum difference 195.55), and it was reverted. The oracle is not
  vacuous.
- Red evidence: compile failures were observed before implementing the
  ternary, sensitivity, selector, planner, diff, generic-width codec, registry,
  and candidate steps. For the original Q4 codec, the tests and the
  implementation were written before the first build, so no separate red run
  exists. After the generalization, its golden-byte tests still pass unchanged.

None of this code is wired into the Qwen2 executor, residency, or Orleans yet.
Every distortion and time value is an estimate, not a measurement on a real
model.

## 2. Precision model: demote what does not matter, restore what does

### 2.1 Identity and state

A precision profile maps each tensor to one encoding. Its `ProfileHash` is the
identity of the effective weights (INV-009) and joins the plan cache, KV, and
prefix cache keys (spec §7.4). Different budgets that yield the same assignment
share one hash. KV produced under one profile is not reused under another
without replay or a compatibility proof.

### 2.2 Importance

- Implemented: the relative output distortion
  `||(W − Q(W))X||² / ||WX||²` on calibration activations. It ranks by effect on
  real activations, not by weight magnitude.
- Next: capture calibration inputs at region boundaries during F1 region
  execution. Measure every tensor of Qwen2.5-0.5B under Q4 and ternary, and
  store an importance map with calibration provenance (TASK-QNT-003). Validate
  the ranking with layer ablations (`SensitivityValidated`) and with held-out
  quality, never on the evaluation split (`HeldOutSplitEnforced`).
- Later: channel/group granularity inside a tensor. Structured salient channels
  stay at higher precision and the rest go low, as in PTQ1.61, BiLLM, and
  PB-LLM. Unstructured masks cost an extra bit or more per weight.

### 2.3 Selection under a budget (implemented)

The selector is greedy by distortion increase per saved byte. Its demotion
order does not depend on the budget, so it is monotone: raising the budget
restores the most important demoted tensors first and never lowers any
tensor's precision (`LargerBudgetOnlyPromotes`). `Diff` lists the exact
promotions or demotions the residency manager must apply. A device without a
kernel for an encoding never receives it. Approximation requires an explicit
session opt-in.

### 2.4 Automatic restore: three mechanisms

1. **Exact escalation through verification (preferred).** The approximate
   profile drafts k tokens, and the source profile verifies the block in one
   pass. A rejected token is replaced by the source output, so committed
   output equals the source model's greedy output. Committed KV always comes
   from the verifier; drafts live in the working branch (ADR-003 state rules,
   F4.1).
   - Precision "returns" automatically at exactly the tokens where the
     approximation matters.
   - The verifier streams its weights once per block, so its bandwidth is
     amortized over k tokens.
   - The acceptance rate per profile is also an online importance signal.
   - Reference: QSpec (arXiv 2410.11305, EMNLP 2025) drafts with W4A4 and
     verifies with W4A16 over shared weights and KV. It reports up to 1.64×
     without quality loss; that is the paper's claim, not a Synapse result. In
     our variant the draft and verify weights differ, so memory is either both
     encodings resident, or a resident draft plus verifier weights paged by
     region.
   - Test: `QuantizedDraftGreedyEquivalent`.
2. **Selective escalation (quality-bounded).** Run the approximate profile
   alone, and recompute a step or block at source precision when an online
   monitor fires (for example a low logit margin or high entropy). It is
   cheaper than mechanism 1 but not exact: a confidently wrong token passes. It
   must be evaluated against mechanism 1 on the sealed suite before it is
   offered.
3. **Memory-driven promotion.** When memory frees up, re-run the selector with
   the larger budget, apply `Diff` promotions (loading source encodings from the
   package or CAS), and give new sessions the better profile. Use hysteresis and
   a cooldown to prevent flapping. Running sessions keep their profile until a
   safe point with replay (spec §6.4). Sessions using mechanism 1 need no replay,
   because their committed state is already at source precision.

### 2.5 Tiers and what BitNet teaches (facts checked 2026-09-28)

- Tiers are not fixed to 4 bits. Any model arrives in its own source encoding
  (F32, F16, BF16, GGUF Q8_0, and later other GGUF quants). On-the-fly targets
  are any `syn.q{2..8}.symmetric.g{8..1024}` member plus `syn.ternary`, so a
  tensor's options can be source → q8 → q6 → q5 → q4 → q3 → q2/ternary. Which
  ladder a model uses is a profile choice, measured per tensor by
  `PrecisionCandidates.Measure`. Residual overlays `W = base + R` (spec §6.3)
  come later.
- Imported source formats decode to the FP32 reference (implemented; see
  `docs/Features/ModelPackages.md`), and may then be re-encoded. An
  unsupported source type fails at import (INV-007), never silently.
- Still missing for "any model":
  - the grid-coded GGUF types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
    IQ3_S), whose ggml lattice tables must come from pinned MIT llama.cpp
    source with attribution, plus the same oracle test;
  - NVFP4 and Q1_0;
  - asymmetric (zero-point) on-the-fly targets;
  - FP8 on-the-fly targets with an explicit scale recipe (spec §4.2);
  - per-family FP8 block-scale application;
  - wiring the decoders into GGUF/SafeTensors tensor reads (they currently
    operate on byte spans).
- BitNet b1.58 is trained ternary from scratch (quantization-aware training
  with per-tensor absmean weights and 8-bit per-token activations). Its
  quality does not transfer to post-training quantization.
- Plain post-training quantization of a trained model to 1-bit or ternary
  collapses quality. BiLLM reports LLaMA-7B WikiText2 perplexity going from
  5.68 in FP16 to 168,388 with 1-bit round-to-nearest. Keeping salient weights
  at higher precision helps a lot (BiLLM 35.04 at 1.09 bits), but gaps remain
  on heavily trained models.
- Therefore ternary in Synapse is (a) a tier that the importance measurement may
  assign only to measured-insensitive tensors or channels, always behind
  mechanism 1 or a quality gate, and (b) a native path for models trained
  ternary.
- `microsoft/bitnet-b1.58-2B-4T` is MIT. It has BF16 master weights
  (`-bf16`), ReLU² FFN, subln, no biases, and the Llama 3 tokenizer. Importing
  the BF16 masters with a per-tensor absmean encoding reproduces the training
  quantizer. That needs a separate `syn.ternary.absmean.tensor.v1` encoding and
  a BitNet family adapter.
- The bitnet.cpp build and conversion flow uses Python, so Synapse reads the
  GGUF or safetensors natively. Its lookup-table kernels (TL1/TL2, from T-MAC)
  are a performance reference for a ternary CPU kernel in the kernel bake-off.

## 3. Concurrent requests

- Region-level continuous batching (TASK-GEN-007): each region keeps a bounded
  queue, and tokens of different sessions that reach the same region share one
  kernel launch (one expert, in MoE). Decode is prioritized over prefill chunks,
  with a registered maximum wait. Tests: `RaggedBatchMatchesIndependent`,
  `LongPrefillDoesNotStarveDecode`, `CancelledSlotReusedSafely`.
- Admission reserves shared weights once and KV per session through the ledger.
  Under pressure, a new session is admitted with a lower profile plus
  mechanism 1, or it waits. Admission never silently degrades a
  non-opted-in session.
- Planner objectives (implemented): `Throughput` for pipelines serving many
  sessions (bottleneck stage), and `Latency` for one sequence (sum of stages and
  links). In the tests the same two devices produce a 9/3 layer split for
  throughput and a single fast device for latency.

### 3.1 Local continuous batching (implemented 2026-09-28, ADR-007)

- The first form is whole-model steps, not yet region queues.
  `Qwen2Model.GenerateAsync` enqueues a session on a dedicated scheduler
  thread. Each session owns an FP32 KV slot, bounded by
  `MaximumConcurrentSessions`.
- Each step puts every decoding session's token first, then FIFO prompt
  chunks. One forward pass serves the whole step. The vocabulary projection
  reads its weights once for every row that needs logits.
- Green tests on the owner's ARM64 Mac and in an amd64 container:
  - `RaggedBatchMatchesIndependent`, for managed, native, and reference;
  - `LongPrefillDoesNotStarveDecode`;
  - `CancelledSlotReusedSafely`;
  - KV parity tests `IncrementalDecodeMatchesFullPrefill`,
    `GenerationRepeatsAfterLongerPrompt`, and
    `LongPromptAcrossChunksMatchesSequential`.
- A local diagnostic on the owner's M2 Pro used five identical 64-token
  requests with the `native` backend. Aggregate output was about 430 tok/s
  against about 150 tok/s for one request at eight threads. At two threads it
  was about 162 against 118. The two-thread gain is small because the batched
  int8 dot is compute-bound there. The next kernel step is register blocking
  across tokens (4 rows × 4 tokens) or `i8mm` on CPUs that report it. These
  numbers are not paired release evidence.
- Remaining for E7:
  - region-level queues;
  - a registered maximum wait;
  - admission through the reservation ledger;
  - a latency SLO for decode under long prefills.

### 3.2 How the batching engine distributes

The topology decision is ADR-009. This follows `flybrain.plan.md` §4 and F6,
and §5 below. It is a hypothesis
list to measure, not a result.

1. **Replicate before splitting.** A model that fits one worker (Qwen2.5-0.5B
   is about 555 MiB resident) runs as independent replicas. Each worker owns
   one model instance and one batching scheduler. The Orleans `SessionGrain`
   routes each new session to a worker, then records only the worker
   incarnation and the epoch. Throughput scales with workers, and steady-state
   decode makes zero grain calls and zero network hops.
2. **Route with affinity.** The router prefers a worker whose ZoneTree prefix
   index already holds the session's prefix KV. After that it prefers the
   worker with the most free KV slots, weighted by its measured
   weight-streaming rate. `WorkerRegistryGrain` publishes that rate per kernel
   (`managed-arm64-sdot`, `native-x64-avx2`, …). Moving KV is expensive, so
   the router moves sessions, not KV.
3. **Pipeline only when needed.** Use a layer-range pipeline (F6.1) only when
   the weights or KV do not fit one worker. The step packet carries the whole
   batch: every token's hidden vector (896 × 4 B for this model) plus
   `(session, epoch, step, position, plan hash)`. Each stage owns the KV of
   its layers for every session in the batch. Two batches in flight keep the
   stages busy. Sampling happens at the last stage, and the sampled tokens
   return to the first stage's scheduler.
4. **Heterogeneous devices.** The planner's `Throughput` objective takes the
   measured per-kernel GB/s and link profiles. A slow x64 node gets fewer
   sessions or fewer layers. It never gets an encoding its kernels cannot
   run.
5. **Failure.** A stale epoch or plan hash is rejected at the receiver. A lost
   worker loses its hot KV. The session re-prefills on a new owner unless a
   durable snapshot exists. Exactly-once is not claimed.

## 4. Loading only what is needed

This follows `flybrain.plan.md` F2 (tensor residency under a budget) and F5
(expert paging). Additions:

- The output of on-the-fly quantization is cached in the local CAS under
  `(source tensor hash, encoding ID, codec version)` and indexed in ZoneTree.
  Each tensor is quantized once and reused across sessions and restarts.
- A promotion is a residency load of the source encoding; a demotion releases
  the source bytes after in-flight fences.

## 5. Heterogeneous Orleans cluster

- Implemented planner: bounded search (spec §10.3); each stage reserves KV for
  the context and gets the highest precision that fits its device memory and
  kernels; typed `InsufficientMemory` and `ApproximationNotAllowed` failures.
- Next inputs:
  - `DeviceProfile` from the hardware profiler (TASK-PLN-001), with a
    weight-streaming rate per encoding (Q4 and ternary kernels differ), dispatch
    overhead, and memory.
  - A measured `LinkProfile` per device pair.
  - Layer tensors from `ModelGraph.Weights` with measured distortions.
- Orleans mapping:
  - `WorkerRegistryGrain` publishes each worker's `DeviceProfile` and its
    supported encodings, i.e. its kernel capability registry.
  - `ModelDeploymentGrain` runs the planner when workers join or leave, and
    stores the plan with `PlanHash` and epoch.
  - `RegionPlacementGrain` holds the leases of each stage.
  - Sessions bind to a plan epoch. Replanning happens only at safe points,
    with hysteresis (spec §10.4).
  - The activation packet (hidden state, position, epoch, plan hash) travels
    worker to worker. No grain is called per token.
- Different architectures (Apple Metal, x64 AVX2, CUDA) differ in supported
  encodings and measured rates. The planner never places an encoding that the
  device cannot execute, and the reference C# path remains available
  everywhere.
- Head or KV sharding (`flybrain.plan.md` F6.4) becomes another stage type
  later.

## 6. Import into the Synapse graph format

This extends ADR-004 (model package sources and import). The pipeline:

```text
GGUF v3 | safetensors + config.json | (later) ONNX subset
    → family adapter (qwen2, llama, olmoe, granitemoe, bitnet)
    → Model IR: regions, operation attributes, WeightDescriptors, fingerprint
    → package: manifest, graph, content-addressed source weight chunks,
      tokenizer/template, importance map (with calibration provenance),
      cached precision variants
```

Rules: no Python in any step; a deterministic conversion identity (spec §5.5);
tensor inventory, tied-weight aliases, and shapes checked; unsupported
operations or encodings fail before generation (INV-007).

## 7. Ordered steps

- [x] E1 Codecs, sensitivity, selector with promotion diff, and heterogeneous
  planner (section 1).
- [ ] E2 Capture calibration activations from F1 region execution; build a
  Qwen2.5-0.5B importance map for Q4 and ternary; validate it with ablations
  (TASK-QNT-003).
- [ ] E3 Q4 and ternary region kernels (reference first) in the executor.
  Test: `MixedProfilePreservesTokensWithinDeclaredTolerance`; report bytes read
  per token.
- [ ] E4 A CAS cache for on-the-fly quantization, and residency integration of
  promotion and demotion through `Diff`, with hysteresis.
- [ ] E5 Exact escalation through verification, after F4.1. Tests:
  `QuantizedDraftGreedyEquivalent`, plus acceptance rate per profile.
- [ ] E6 Selective escalation, evaluated against E5 on the sealed suite.
- [ ] E7 Region-level continuous batching and admission with per-session
  profiles (TASK-GEN-007).
- [ ] E8 Planner fed by measured device and link profiles with per-encoding
  rates; the Orleans deployment grain; two processes, then two nodes.
- [ ] E9 Importers: GGUF llama and olmoe, safetensors SmolLM2, and BitNet 2B4T
  BF16 native ternary with a per-tensor encoding.
- [ ] E10 Structured salient-channel mixing inside tensors, and residual
  overlays (TASK-PRE-001..002).

## 8. Do not

- Do not claim quality for post-training ternary without the sealed evaluation.
- Do not change the effective weights of a running session without replay or
  verification.
- Do not let an approximate profile commit state in exact mode; only the
  verifier commits.
- Do not run bitnet.cpp's Python tooling.
- Do not call a grain per token, head, or region activation.
