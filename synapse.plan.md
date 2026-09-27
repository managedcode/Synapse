# Synapse master plan

Status reflects executed evidence only. The design specification remains the
normative product scope.

## Bootstrap sequence

- [x] Record product constraints, upstream roles, and initial ownership.
- [x] Inspect current dotLLM, LLamaSharp, ZoneTree, .NET, Rust, and Aspire
  revisions without claiming benchmark results.
- [x] Make macOS ARM64 the primary local/GitHub runner target and keep Linux
  x64 as the secondary portable CPU gate.
- [ ] Complete `TASK-BST-002`: locked restore, dependency audit, and repeatable
  clean restore evidence.
- [ ] Complete `TASK-BST-003`: C#/Rust workspace, offline doctor, typed budget
  failure, ZoneTree durability probe, and non-zero real tests.
- [ ] Complete `TASK-BST-004`: one C# task runner with child-process evidence,
  cancellation, timeout, and zero-test rejection.
- [ ] Complete `TASK-BST-005`: CI, coverage, architecture, and hardware matrix.
- [ ] Complete `TASK-BST-006`: pinned repo-local MCAF/.NET skills and discovery.

## Inference critical path

- [x] Bounded GGUF v3 header/index plus managed Qwen2 Q8_0 first-token
  execution matching dotLLM and LLamaSharp on the pinned fixture.
- [ ] Complete general model package/compiler work. The pinned multi-family
  catalog, safe downloader, bounded SafeTensors index, and GGUF adapter exist;
  the Synapse manifest/chunk format, tokenizer/template import, and bounded
  allocator do not.
- [x] Typed Model IR core, bounded shapes, explicit numerical types, FlyBrain
  regions/eligibility, and verifier for SSA/order/cycles/state/entry points.
- [x] Materialize and verify the managed Qwen2 dense topology as Model IR before
  runtime scratch/KV allocation.
- [ ] Execute Qwen2 from the IR, add Execution IR/lifetime planning, and
  complete the scalar numerical oracle.
- [ ] C# paged KV ownership, rollback, and ZoneTree prefix metadata.
- [ ] Repo-owned tokenizer plus pinned SmolLM/Qwen import and text generation.
- [ ] .NET SDK, worker IPC, bounded streaming, and cancellation.
- [x] Real-process smoke runner against pinned dotLLM and LLamaSharp baselines.
- [ ] Complete the statistical paired benchmark runner, evidence validator,
  and verdicts. One 3-warm-up/5-measurement interleaved Mac run with raw
  samples exists; the required 30 paired samples and verdict logic do not.
- [ ] CPU SIMD, Q4, Metal, and measured quality-preserving optimization; use
  Rust only for a profiled managed hotspot.
- [ ] Aspire/Orleans control plane, direct worker data plane, fencing, and
  multi-process then two-node recovery evidence.

## Benchmark release gate

No performance task is complete until the same model, tokenizer/template,
prompt token IDs, context/output limits, sampling, hardware, and durability
semantics are recorded for Synapse, dotLLM, and LLamaSharp. The required win is
evaluated from raw paired runs; missing/incompatible baselines are
`inconclusive` or `ineligible`, never a pass.

## 2026-09-28 working checkpoint

- [x] Replace repository model storage with a source-controlled catalog and
  ignored, content-verified local downloads. Current sets cover Qwen2, Llama,
  Qwen3, Mamba, Phi3, distilled Qwen2, Mistral3, and two BERT embedding models.
- [x] Reject unsafe package/file paths, HTTP or unrelated redirect hosts,
  more than five redirects, incorrect declared size, streaming overflow, and
  SHA-256 mismatch; publish downloads only after verification.
- [x] Verify real SafeTensors indexes for SmolLM2, MiniLM, and BGE, and keep
  architecture dispatch independent of model branding.
- [x] Add locked 10-turn dialogue and embedding retrieval scenarios.
- [x] Record a real equal-12-thread Qwen2 smoke comparison with load, TTFT,
  decode, CPU, wall, and memory samples. Energy is explicitly
  `not_run_missing_privilege`.
- [ ] Remove the old Qwen GGUF object from reachable Git history, verify no
  model blob remains, and force-with-lease the rewritten branch.
- [ ] Turn the benchmark evidence JSON into the validated `TASK-BMK-001`
  schema/runner; add thread scaling, 30 paired runs, locked/live 10-turn,
  embeddings, cache states, and direct llama.cpp.
- [ ] Execute Qwen2 from the verified region IR instead of the parallel shadow
  loop, following `flybrain.plan.md` F0/F1 and accepted ADR-003.
