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
- [ ] General model package validation and bounded allocator.
- [ ] Typed Graph IR, verifier, scalar operators, and tiny deterministic model.
- [ ] C# paged KV ownership, rollback, and ZoneTree prefix metadata.
- [ ] Repo-owned tokenizer plus pinned SmolLM/Qwen import and text generation.
- [ ] .NET SDK, worker IPC, bounded streaming, and cancellation.
- [ ] Native benchmark runner against pinned dotLLM and LLamaSharp baselines.
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
