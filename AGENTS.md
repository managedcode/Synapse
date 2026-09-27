# Synapse repository instructions

## Start here

Read `docs/Architecture.md`, `synapse.plan.md`, the applicable
`docs/Features/<SliceName>.md`, its ADR, and the selected task in
`docs/Development/tasks.json`. Read the nearest project-local `AGENTS.md`
before editing. Never mark planned runtime or benchmark work as complete.

## Non-negotiable product rules

- Product code and tooling use C#/.NET and Rust. Native Metal/CUDA kernels are
  allowed. Python and Node.js are prohibited build, runtime, import,
  conversion, calibration, training, test, benchmark, and tooling
  dependencies.
- Keep one monorepo and the same PascalCase `Features/<SliceName>` under every
  applicable technical root.
- Orleans accepts and coordinates distributed requests as the coarse control
  plane. C# owns the portable/reference path; Rust owns an optimized kernel,
  allocator, hot-KV operation, or direct transfer path only after profiling
  proves the boundary. Local mode must work without Orleans or a network.
- Model math, precision, adapters, and KV compatibility are explicit. Do not
  skip dense layers without a qualified numerical or routing profile.
- Session state has one fenced owner. Stale epochs and conflicting retries
  fail. Do not claim exactly-once delivery without end-to-end evidence.
- The scalar reference backend and explicit local execution remain permanent,
  tested capabilities. Unsupported optimized paths must fail explicitly.
- ZoneTree is the required embedded durable store for cache metadata, prefix
  indexes, journals, and related local state. Do not duplicate or replace it.
- dotLLM and LLamaSharp are external benchmark subjects, never hidden runtime
  engines. dotLLM is GPL-3.0: inspect and execute it as a separate baseline,
  but do not copy its source into this MIT repository.
- Never fabricate benchmark numbers, successful tests, hardware availability,
  upstream installs, or delivery state.

## Workflow

For non-trivial work, update the applicable root `*.brainstorm.md` and
`*.plan.md`. Work in dependency order: one plan step, a failing behavior test,
the implementation, the green related suite, a quality pass, and a factual
plan update. Every behavior change maps stable REQ/AC/TASK/TEST identifiers.
Update an ADR before changing a public contract, schema, state model,
dependency, topology, or trust boundary.

Do not use worktrees for this repository unless the user explicitly asks.
Preserve unrelated changes. Use the current checkout.

## Code and tests

- C#: `net10.0`, nullable enabled, warnings as errors, Central Package
  Management, async cancellation across boundaries, no sync-over-async.
- Rust: pinned stable toolchain, edition 2024, `Result` for expected failures,
  bounded documented unsafe, no panic across ABI or protocol boundaries.
- TDD is mandatory. Verification uses real components and processes; mocks,
  fakes, stubs, and service doubles are prohibited. Deterministic tiny model
  fixtures are allowed because they execute real math.
- A required suite with zero tests fails. Missing hardware is
  `not_run_missing_hardware`, never `passed`.
- Default limits: file 400 LOC, type 250 LOC, function 60 LOC, nesting depth 4.
- Run format, build, analyzers, architecture checks, security checks, coverage,
  and focused regressions before marking a task verified.

## Canonical commands

```text
dotnet restore Synapse.slnx --locked-mode
dotnet build Synapse.slnx --configuration Release --no-restore
dotnet test Synapse.slnx --configuration Release --no-build
cargo test --manifest-path native/Cargo.toml --locked
cargo fmt --manifest-path native/Cargo.toml --all --check
cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings
```

Use `tools/Synapse.Build` once its task runner is implemented. Long-running
tests must show progress. Store exact commands, exits, and limitations in task
evidence. A benchmark claim requires raw paired evidence and quality parity.
