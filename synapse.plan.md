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
  region activation/provenance/skip contracts, typed operation attributes and
  decode position, and
  verifier for SSA/order/cycles/state/entry points. Model-bounded symbolic KV
  context and canonical graph fingerprinting keep identity session-independent;
  weight descriptors bind tensors to verified source ranges and encodings.
  F0.7 adds decision-value, non-causal route, bypass, tolerant-merge, and
  position-hole verifier rules; 13 focused region tests and the full 54-test
  real-model Mac gate pass. Execution of conditional regions remains F1.
- [x] Materialize and verify the managed Qwen2 dense topology as Model IR before
  runtime scratch/KV allocation. Entry plumbing is outside regions; the
  verifier independently derives exact value, tensor, and state boundaries.
- [ ] Execute Qwen2 from the IR, add Execution IR/lifetime planning, and
  complete the scalar numerical oracle. Initial C# reference linear/bias,
  causal GQA, RMSNorm, SiLU, element-wise, Softmax, and RoPE operators now have
  FP64 oracle/masking tests. A first tiny fixed-shape dense graph interpreter
  executes from verified IR with declared FP64 accumulation and fail-closed
  preflight; 68/68 local Release tests pass. The Qwen graph interpreter,
  scheduler, and remaining ops are open.
- [ ] C# paged KV ownership, rollback, and ZoneTree prefix metadata.
- [ ] Repo-owned tokenizer plus pinned SmolLM/Qwen import and text generation.
  Qwen2 byte-level BPE from the GGUF arrays is implemented and matched
  `llama-tokenize` on the whole repository corpus (ADR-014). Other
  pre-tokenizers fail explicitly. `generate` reports decoded text.
- [ ] .NET SDK, worker IPC, bounded streaming, and cancellation.
- [x] Real-process smoke runner against pinned dotLLM and LLamaSharp baselines.
- [ ] Complete the statistical paired benchmark runner, evidence validator,
  and verdicts. One 3-warm-up/5-measurement interleaved Mac run with raw
  samples exists; the required 30 paired samples and verdict logic do not.
- [ ] CPU SIMD, Q4, Metal, and measured quality-preserving optimization; use
  Rust only for a profiled managed hotspot. In progress in
  `cpu-kernels.plan.md` (ADR-006): the managed SIMD and Rust Q8_0 kernels,
  worker pool, fused regions, batched prefill, and explicit
  `reference|managed|native` backends are implemented and tested locally. Q4,
  Metal, and remote CI evidence are still open.
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
- [ ] `TASK-BMK-001` direct-native checkpoint: add a failing real-process
  Qwen parity test, implement a pinned CPU `llama-completion` subject with
  explicit timing provenance, make macOS/Linux CI build and exercise it, record
  raw short/long diagnostics, and show honest coverage for every catalog model.
  This checkpoint does not close the statistical benchmark task.
- [ ] `TASK-BMK-001` expansion after native GGUF control: qualify a no-Python
  MLX Swift/native Mac Metal subject and a C# ONNX Runtime GenAI subject with
  pinned packages, architecture/tokenizer/quality checks, and separate
  hardware/precision cohorts before any cross-engine timing claim.
- [ ] `TASK-BMK-001` workload expansion: verify 128-token single-request and
  three 64-token locked-transcript diagnostics on all four CPU subjects, keep
  early EOS and quality-review status visible, and publish raw evidence and
  summaries in performance CI only. A separate resident-session implementation
  is required before reporting KV/prefix-cache hit and miss performance.
- [ ] `TASK-BMK-001` memory checkpoint: test the absence of a whole-process
  memory envelope, then record comparable pre-load-through-exit peak resident
  bytes and macOS physical footprint for every external process. Expose CLR
  heap/allocated diagnostics only for instrumented managed subjects; keep
  private virtual size and missing metrics explicitly distinct. Do not mark
  complete until real-process regressions and raw paired evidence pass.
- [ ] `TASK-BMK-001` diagnostic matrix checkpoint: a C# process runner executes
  Synapse, dotLLM, LLamaSharp, and direct llama.cpp in rotated fresh-process
  rounds, verifies the same pinned continuation, and writes immutable raw
  JSON with per-subject whole-process memory. This remains a smoke diagnostic
  until 30 paired measurements, locked dialogue, and schema validation land.
- [ ] `TASK-BMK-002` Foundry Local subject (ADR-011): failing TUnit checks for
  the runner-memory plan, invalid sets, the no-download refusal, real
  streaming evidence, the separate-cohort report, and workflow isolation; then
  the RID-isolated SDK 2.0.1 project, the Qwen/Phi/Mistral/DeepSeek model set,
  a real local Mac run, and one CI job per runner and model. Do not mark
  complete until the hosted jobs have run and uploaded raw evidence.

Foundry Local checkpoint (2026-09-28): the eight new TUnit checks failed
first, then passed; the later context-bound and system-prompt checks also
failed before their implementation. A probe showed that every Foundry package
preallocates FP32 KV for its full context window (Phi-3.5-mini: 98 GiB
compressed footprint, 12.4 s first token), so the set caps `max_length` at
1,024 and keeps both configs hashed. Mistral 7B v0.2 has no system role and
uses `prepend-to-first-user`. Local Mac evidence covers all six CPU models and
the Qwen2.5 WebGPU variant for the 128-token and three-turn scenarios (14 raw
files plus two package-default probes). The full .NET suite passed 189/189
with the local dotLLM and Homebrew llama.cpp `b29c606e2` subjects. The 15
hosted Foundry jobs and the `verify.yml` anchor fetch have not run.

Public-results checkpoint (2026-09-28): README and benchmark documentation
claims are now scoped to the raw evidence. The native CPU result is presented as an
eight-token diagnostic (102% of llama.cpp at two threads, 114% at eight), the
older long-prompt gap is not attributed to the new batched-prefill build, and
the manually reviewed Foundry outputs are reported as quality-ineligible: no
model/device variant fully completed the 128-token instruction. Performance
diagnostics now trigger on every push to `main` as well as manual dispatch.
This does not close `TASK-BMK-001`, `TASK-BMK-002`, or any CPU task; fresh
hosted evidence is still required.

Hosted aggregation follow-up (`TASK-BMK-001`, `AC-BMK-001-4`): run
`36435838291` completed its 20 jobs and published 22 raw artifacts, but no
run-level results table. Add a final C# artifact collector and Actions job;
verify its partial-evidence behavior first, then render the actual 22-artifact
run and check the new hosted reporting job. The long-output quality gate and
30-pair release verdict remain open.
Local checkpoint: the aggregate command processed all 22 raw artifacts from
that run into 124 result rows with no missing evidence; the two focused
real-evidence regressions passed, and `actionlint` accepted the workflow.
The new final Actions job is not hosted-verified until its next run finishes.

Local direct-native evidence: pinned Homebrew llama.cpp `b29c606e2` reproduced
the Qwen prompt token IDs and continuation, three focused real-process native
regressions passed, and raw 8/128-token samples plus a separate `llama-bench`
control are recorded. The full local 107-test suite passed before concurrent
ModelPackages tests were added; the later 130-test run reported all test cases
successful but the process exited 134 while loading ggml's in-process oracle.
No benchmark task is verified, and the new CI native path has not run remotely.

Whole-process memory/matrix checkpoint: the missing-field and missing-runner
real-process tests failed first, then passed after implementation. A 3+5
rotated Mac CPU run retained 32 raw Qwen samples for all four subjects,
including RSS, macOS footprint, and CLR diagnostics where instrumented.
A separate 32-token run is `ineligible_quality_mismatch` because dotLLM
diverges from the LLamaSharp/native continuation; Synapse has token IDs but no
owned text decoder yet. The current full .NET gate passes 134/134, while the
earlier ggml exit 134 has not been conclusively explained. Coverage collection
passes but the aggregate line rate is 70.2% and child-process CLI execution is
not attributed to the test-host report; the required changed-line coverage
gate and remote CI are not verified. No `TASK-BMK-001` release claim is closed.

CI separation follow-up: Ubuntu `verify` on `3bbd5ee` ran 73 tests with one
failure because a functional assertion required a platform-specific 32-token
dotLLM divergence; the same test timed out after five minutes on hosted macOS.
The measured matrix is now removed from TUnit entirely: tests use recorded raw
evidence for report and quality-gate behavior. Performance measurements are
in a distinct manual Mac/Linux/Windows Actions workflow with raw artifacts
and a C# report.
The report behavior test was red (unknown `report` command) before the renderer
was added; the quality gate was separately red (usage exit 2) before its exit-3
implementation. The first three-OS performance run (`9567d06`) passed and
published three raw artifacts; Windows and Ubuntu verify passed while hosted
Mac verify was still running at this checkpoint. The quality-gate follow-up
and statistical release gate remain unverified until new runs complete.
- [ ] Execute Qwen2 from the verified region IR instead of the parallel shadow
  loop, following `flybrain.plan.md` F0/F1 and accepted ADR-003.
- [ ] CPU kernel and concurrency checkpoint (`TASK-CPU-001..005`; ADR-006 and
  ADR-007).
  - Implemented:
    - managed `Dp`/AVX2/portable kernels and the Rust `synapse-kernels`
      cdylib;
    - a persistent worker pool, fused regions, and batched prefill;
    - an embedding-excluding page prefetch and allocation-free GGUF metadata
      skipping;
    - continuous batching of concurrent requests over per-request KV slots.
  - The scalar path remains the `reference` backend.
  - Local gates pass on ARM64. The x64 AVX2 managed and native paths passed
    their suites in a `linux/amd64` container. AVX-VNNI has not run on
    hardware, and no hosted Actions run of this change set exists.
  - Every backend matches the eight-token continuation. At 32 tokens the
    Q8-activation paths diverge at token 22, where the FP32 reference's own
    top-2 margin is 0.0115 logits.
  - ADR-009 fixes the Orleans cluster direction; it is not implemented.
- [ ] GPU and long-context checkpoint (`gpu-kernels.plan.md`; ADR-012, ADR-013;
  `TASK-GPU-001..004`, `TASK-CTX-001..003`).
  - Implemented:
    - explicit context limits and a YaRN profile, plus `--tokens-file`,
      `--rope-scaling`, and prefill progress;
    - the brand-neutral `synapse-gpu` ABI and a Metal dense-decoder backend:
      matrix-vector, GEMM, flash attention, and split decode;
    - one backend-neutral step encoder shared by Metal and CUDA;
    - watchdog-safe submissions for long prompts;
    - CUDA as code only (driver API plus NVRTC).
  - Verified locally on the M2 Pro:
    - 10 Metal TUnit tests and 1 CLI Metal test;
    - 11 Rust `synapse-gpu` tests;
    - 5 context tests and 2 CLI long-prompt tests.
  - The CUDA kernels only passed a C++ syntax check with a local shim. They
    have not been compiled by NVRTC or run on NVIDIA hardware.
  - Raw long-context evidence and the pass-key quality run are still open.
