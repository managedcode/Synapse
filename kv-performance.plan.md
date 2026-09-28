# Hot KV performance selection plan

Date: 2026-09-28. Status: plan only; no candidate is implemented or
measured. Parent plan: `flybrain.plan.md`. This follows the accepted D1 rule:
C# is the reference path, and Rust takes hot KV only when paired profiling
justifies it.

Owner direction: use whichever KV implementation performs best, decided by
measurement rather than by language preference.

## 1. Two different things are called "KV"

| | Hot attention KV cache | ZoneTree key-value store |
|---|---|---|
| What it holds | Key/value tensors of every past token, per layer and KV head | Durable ordered records |
| Access pattern | Every decode step reads all past K/V of every layer | Point lookups, range scans, appends through WAL and compaction |
| Volume | Qwen2.5-0.5B in FP32 is 24 layers × 128 KV width × 2 × 4 B = 24 KiB per token, so a 4,096-token context reads about 96 MiB per generated token | Metadata is small; payloads depend on use |
| Bound by | Memory bandwidth and the attention kernel | Disk I/O, serialization, compaction |
| Role in Synapse | Session-owned memory that the attention kernel reads directly | Prefix index, page metadata, snapshot catalog, commit ledger, and the cold tier candidate below |

ZoneTree is the required store for durable KV metadata and a real candidate
for the cold tier (spilled or persisted pages). For the hot tier it is measured
once as control C0, so that the decision rests on data. It is not expected to
compete there, because the hot path must be plain memory that the kernel
streams.

## 2. Why "C# or Rust" is the last question, not the first

Current facts:

- The C# `Qwen2KvCache` is one flat FP32 array per layer in position-major
  order (`[position][kvWidth]`). For one head, consecutive keys are
  `kvWidth` floats apart.
- The C# `Qwen2Attention` loops over all 14 query heads. For each head it
  re-reads every cached key and value of its KV head, and it accumulates values
  in scalar code. With Qwen2.5-0.5B grouped-query attention (14 query heads, 2
  KV heads), every K/V row is therefore read 7 times per layer per step.
- The Rust `PagedKvCache` uses one `Vec` per 16-token page and returns a
  checked `Result` for every head read. The executor does not use it.

Comparing these two as they are would measure layout and checking
differences, not the language. The hypotheses below are ordered by expected
impact. They are to be measured, not claimed:

1. Group-aware attention reads each K/V row once for all query heads in the
   group.
2. A head-major page layout (`page[kvHead][token][headDim]`) makes the
   attention stream contiguous memory.
3. A fused SIMD paged kernel with online softmax works in one pass over the
   pages.
4. FP16/BF16 KV storage with FP32 accumulation halves the bytes read. This is a
   separate numerical profile; spec §7.3 names FP16/BF16 as the initial KV
   release.
5. Parallelism across KV heads or page ranges helps at long context. This is
   the local form of attention-head sharding. Its cross-worker form, with
   Orleans placement, is `flybrain.plan.md` F6.4, and the KV contract here
   must allow one store per head group or position range.
6. Language: C# `Vector128`/`AdvSimd` versus Rust NEON behind a C ABI.

## 3. Shared contract and conformance suite

Every candidate implements the same semantics, and timing counts only after
the full conformance suite passes against the real component. A Rust candidate
is exercised from the C# test project through the real FFI boundary.

- Paged storage with page tokens P in {16, 32, 64}.
- Single-owner append with owner generation.
- A per-layer valid-position bitmap; holes are allowed only for slots with
  `PositionHolesAllowed` (ADR-003).
- `BeginBranch` (copy-on-write), `CommitPrefix(k)`, and `RollbackTail`
  (spec §7.5).
- A bounded page pool reserved through the memory ledger, with no allocation
  per decode step.
- A device-agnostic page table, so a later Metal backend can own pages.

Tests: `KvAppendRequiresOwner`, `KvRollbackRestoresPrefix`,
`KvBranchCopyOnWrite`, `KvHolesOnlyWhereAllowed`,
`KvPoolBoundedNoStepAllocation`, `AttentionMatchesReferenceInterpreter`
(against the F1.1 scalar oracle within the declared tolerance), and
`AttentionMasksHoles`.

## 4. Candidates

Hot tier:

| ID | Storage | Attention kernel | Purpose |
|---|---|---|---|
| K0 | Current C# flat FP32 | Current scalar C# | Frozen baseline |
| K1 | C# paged, head-major, FP32 | C# group-aware fused SIMD | Best managed candidate |
| K2 | K1 with FP16 storage | K1 with FP32 accumulation | Separate numerical profile |
| K3 | Rust paged pool, same layout | Rust kernel, same algorithm; one C-ABI call per attention region per step | Rust candidate |
| K4 | C#-owned native memory (`NativeMemory.AlignedAlloc`) | Rust kernel over that memory | Separates storage owner from kernel language |
| C0 | ZoneTree values keyed by (layer, kvHead, page) | K1 kernel after lookup | One-off control |

Cold tier, decided separately:

| ID | Payload | Index |
|---|---|---|
| Z1 | Page blobs stored as ZoneTree values | ZoneTree |
| Z2 | Append-only content-addressed page segment files | ZoneTree |

## 5. Measurement protocol (registered before the first run)

- Hardware: the dev Mac recorded by the profiler (exact SoC, RAM, OS, power
  mode, thermal state).
- Model: pinned Qwen2.5-0.5B-Instruct Q8_0 with identical weights for every
  candidate; only KV and attention change.
- Contexts: 512, 4,096, and 16,384 filled positions, then 64 greedy decode
  steps. First one session, later four concurrent sessions.
- M1, microbenchmark: one attention region at context N; 10 warm-up and at
  least 30 measured iterations; randomized candidate order; paired runs.
- M2, end to end: decode through the F1 region scheduler, with the attention
  region kernel swapped. Report p50 and p95 step latency, tokens per second,
  the attention share of step time, KV bytes read per token, allocations per
  step (must be 0), resident KV bytes, and append, rollback, and branch costs.
- Cold tier: spill throughput, restore latency for a 4,096-token prefix, bytes
  written to disk per payload byte, and correctness after a crash and reopen,
  all at the same fsync semantics.
- Parity first. FP32 candidates must produce greedy tokens identical to K0.
  K2 must stay within the declared logits tolerance and produce identical
  64-token greedy output on the prompt set. A mismatch makes the candidate
  ineligible.
- Raw events and the manifest go to `artifacts/benchmarks/<RunId>/`
  (spec §15.1).

## 6. Decision rule (fixed now)

- Primary metric: M2 decode tokens per second at context 4,096.
- A Rust candidate (K3 or K4) replaces the best C# candidate only if the lower
  bound of the 95% CI of its improvement is at least 10%, and no other
  registered context regresses by more than 3%. Otherwise C# stays: one
  language, no FFI, simpler debugging.
- If Rust wins, K1 stays as the permanent reference path (D1), Rust becomes
  the default optimized path, and K0 is deleted.
- If C# wins, the Rust `PagedKvCache` is deleted unless a scheduled Rust
  worker task (TASK-TRN-002) needs it. No idle second implementation remains.
- K2 (FP16) is adopted as a default profile only with parity evidence. FP32
  remains available.
- Cold tier: Z1 or Z2, chosen by restore latency at equal durability.
  ZoneTree holds the index either way.
- Record the verdict and raw data references in
  `docs/Features/SessionState.md` and in ADR-006 (hot and cold KV tiers).

## 7. Steps

- [ ] KV.1 Contract and conformance suite (section 3), run against K0 where it
  applies. The K0 gaps (branching, holes, pool) are recorded, not patched.
- [ ] KV.2 K1 in C#, passing the conformance suite and FP32 parity.
- [ ] KV.3 M1 microbenchmark for K0 and K1. This isolates the layout and
  group-aware effects before any language comparison.
- [ ] KV.4 K3 in Rust behind a C ABI: `Result` mapped to typed error codes, no
  panic across the boundary, documented bounded `unsafe`. It must pass the
  same suite from C#.
- [ ] KV.5 M1 for K1 and K3 (and K4 if K3 is close). Run C0 once.
- [ ] KV.6 M2 end to end, after F1.3 makes the attention region kernel the swap
  point.
- [ ] KV.7 K2 FP16 profile, with parity and M2.
- [ ] KV.8 Cold tier Z1 against Z2.
- [ ] KV.9 Apply the section 6 decision, delete the losers, and write ADR-006.

Ordering: KV.1–KV.5 can run in parallel with F1.2–F1.4. KV.6 needs F1.3. The
verdict must land before F4.1 (the working branch) and F6 (each region owner
keeps its KV), because both build on the chosen store.
