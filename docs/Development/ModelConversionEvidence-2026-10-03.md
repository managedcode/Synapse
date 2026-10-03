# ModelConversion local evidence — 2026-10-03

Scope: `REQ-CNV-001`, `AC-CNV-001-1..5`, `TASK-CNV-001`,
`TEST-CNV-001-1..5`, ADR-024. Current checkout; unrelated changes preserved.
No worktree, stash, dependency addition, commit, push or deployment.

## Behavior-first evidence

Initial converter/CLI cases failed before implementation (exit 2). Initial
pipeline cases failed 7/7, SafeTensors cases failed 24/24; final default-analyzer
green suites passed 13 and 24 respectively. ONNX import/cancellation tests
failed before implementation; additional weight-amplification and null-bound
regressions failed before repairs, followed by 13 green tests. Some initial red
runs disabled analyzers while independent new files were being edited; final
green runs used default analyzers.

Final arity review added
`MatMulRejectsEmptyThirdInputWhileGemmAllowsOmittedBias`. Its real protobuf
MatMul case failed with exit 2 before restricting optional bias trimming to
Gemm. Default-analyzer `Onnx*` regressions then passed 14/14 (exit 0), including
real omitted-bias Gemm results `[7,10]`.

Three real cancellation regressions failed with default analyzers (exit 2):
the scalar interpreter ignored its timer token and completed the row in about
14.5s; conversion checked it only after the row in about 15.2s; a precancelled
interpreter entered preflight instead of failing on cancellation. The valid
row has 4,096 Linear nodes sharing one 1,000x1,000 matrix, a one-second real
timer and a three-second responsiveness assertion. No service double is used.
After adding interpreter cancellation boundaries and token forwarding, direct
Release execution passed 3/3 in 2.385s, GraphExecution 42/42 in 333ms and
ModelConversion 67/67 in 3.314s (exit 0, zero skipped). HTML reports are under
`artifacts/model-conversion/`. A transient MSBuild child-node exit and a
zero-test wrapper exit 5 were not counted as passes; direct discovery and
execution confirmed the real tests.

The native package suite failed its duplicate input-name and duplicate
manifest-property cases before enforcing strict duplicate rejection. The
undeclared ONNX bound case failed independently before rejecting unused bound
names. Tiny fixtures are real protobuf/SafeTensors/GGUF files and real scalar
math; CLI verification launches real processes.

## Commands and results

| Command/check | Local result |
|---|---|
| `dotnet restore Synapse.slnx --locked-mode` | Exit 0 |
| `dotnet build Synapse.slnx --configuration Release --no-restore` | Exit 0, zero warnings/errors; final clean snapshot also used `--no-incremental` |
| `dotnet build Synapse.slnx --configuration Release --no-restore --no-incremental --disable-build-servers -m:1` | Final repaired build: exit 0, zero warnings/errors, 27.24s |
| `dotnet format Synapse.slnx --no-restore --verify-no-changes` | Exit 0 |
| `dotnet test tests/Synapse.IntegrationTests/Synapse.IntegrationTests.csproj --configuration Release --no-restore -- --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.ModelConversion/*/*'` | Exit 0; 63/63, zero skipped |
| `dotnet tests/Synapse.IntegrationTests/bin/Release/net10.0/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.ModelConversion/*/*' --report-html-filename /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/model-conversion-cancellation-green.html` | Exit 0; final repaired scope 67/67, zero skipped |
| `dotnet tests/Synapse.IntegrationTests/bin/Release/net10.0/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.GraphExecution/*/*' --report-html-filename /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/graph-execution-cancellation-green.html` | Exit 0; 42/42, zero skipped |
| `cargo fmt --manifest-path native/Cargo.toml --all --check` | Exit 0 |
| `cargo test --manifest-path native/Cargo.toml --locked` | Exit 0; 32 actual tests: 1 core, 2 bootstrap, 22 GPU-plan/probe, 7 CPU; empty unit/doc targets do not count as required suites |
| `cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings` | Exit 0 |
| `dotnet list Synapse.slnx package --vulnerable --include-transitive` | Exit 0; no reported vulnerable packages in six projects from NuGet |
| `git diff --check` | Exit 0 |
| Architecture/quality review | No new dependencies, hidden engines or prohibited tool languages; same ModelConversion slice under CLI/runtime/tests; native rows lower to verified Model IR; files/types remain within configured limits |

Rust commands use `PATH=/opt/homebrew/opt/rustup/bin:$PATH`; bare `cargo` is not
on this shell's PATH. Sandbox named-pipe/feed failures were rerun with the
required local IPC/network permissions. An early full test lacked the native
baseline version variable and was cancelled. Overlapping builds and coverage
instrumentation caused stale/rewritten DLLs in later attempts; final full-suite
verification uses a separately copied clean Release build.

Baseline environment: `SYNAPSE_DOTLLM_EXECUTABLE` points to
`_external/dotLLM/src/DotLLM.Cli/bin/Release/net10.0/DotLLM.Cli`, with verified
commit `d88040451d7db56e5dfef9d5754ad0955b0f7fe5` in `SYNAPSE_DOTLLM_VERSION`.
`SYNAPSE_LLAMACPP_EXECUTABLE=/opt/homebrew/bin/llama-completion` reports 0.4.1,
build 10964, commit `b29c606e2`, used in `SYNAPSE_LLAMACPP_VERSION`.
`SYNAPSE_MODEL_ROOT` and `SYNAPSE_FOUNDRY_CACHE` point to the existing
`artifacts/models` and `artifacts/foundry-local` under this checkout. No model
download was needed for this slice.

## Coverage and full suite

```text
dotnet artifacts/model-conversion/test-snapshot/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.ModelConversion/*/*' --coverage --coverage-output-format cobertura --coverage-output /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/coverage.cobertura.xml --results-directory /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/focused-tests
```

Exit 0; 63/63. C# XML analysis reports 995/1,114 executable lines (89.3%) in
source files under `Features/ModelConversion`. This is file-scoped coverage,
not the repository changed-line gate. Total solution coverage from this
focused filter is 35.3%. Raw local reports are ignored under
`artifacts/model-conversion/`.

Final repaired coverage command:

```text
dotnet artifacts/model-conversion/final-coverage-snapshot/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.ModelConversion/*/*' --coverage --coverage-output-format cobertura --coverage-output /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/final-coverage.cobertura.xml --results-directory /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/final-focused-tests
```

Exit 0; 67/67, zero failed/skipped, 5.837s. File-scoped converter coverage is
still 995/1,114 executable lines (89.3%); total solution coverage from this
focused filter is 35.4%.

The first clean snapshot full run used `--output Detailed --timeout 15m`.
It ended with exit 3 after 15m 07s: 354 succeeded, zero failed/skipped. The
global timeout cancelled an active CPU prefix-reuse case and queued tests;
this aborted run is not a full-suite pass.

The 35-minute replay was intentionally cancelled after a final review found
the MatMul arity and within-row graph cancellation edge cases. It recorded
282 succeeded, zero failed/skipped, exit 3, in 2m 03s. It is not a full-suite
pass. The fresh snapshot after those repairs uses:

```text
SYNAPSE_DOTLLM_EXECUTABLE=/Users/ksemenenko/Developer/Synapse/_external/dotLLM/src/DotLLM.Cli/bin/Release/net10.0/DotLLM.Cli SYNAPSE_DOTLLM_VERSION=d88040451d7db56e5dfef9d5754ad0955b0f7fe5 SYNAPSE_LLAMACPP_EXECUTABLE=/opt/homebrew/bin/llama-completion SYNAPSE_LLAMACPP_VERSION=b29c606e2 SYNAPSE_MODEL_ROOT=/Users/ksemenenko/Developer/Synapse/artifacts/models SYNAPSE_FOUNDRY_CACHE=/Users/ksemenenko/Developer/Synapse/artifacts/foundry-local dotnet artifacts/model-conversion/final-snapshot/ManagedCode.Synapse.IntegrationTests.dll --output Detailed --timeout 35m --results-directory /Users/ksemenenko/Developer/Synapse/artifacts/model-conversion/final-full-tests
```

Final result: exit 0, 429/429 succeeded, zero failed/skipped, 6m 53.740s.
Raw output is `artifacts/model-conversion/final-full-tests.log`; the real test
report is `artifacts/model-conversion/final-full-tests/ManagedCode.Synapse.IntegrationTests-macos-net10.0-report.html`.
No builds or coverage instrumentation ran against the frozen snapshot while
it executed. Before starting, the snapshot and current clean benchmark DLLs
both had SHA-256
`b53be7320c512ecae6e37fb019c6c6b65e4bbe48b210f399b6a99327a9596c80`.
Final full-solution formatting and diff checks also exited 0.

The final full-run log contains two unobserved `ObjectDisposedException`
messages from `ZoneTree.Core.ZoneTreeMaintainer.StartPeriodicTimer` after
`ZoneTreeDependencyPersistsAcrossReopen` passed. Read-only inspection of the
actual installed ZoneTree 1.9.8 DLL supports an existing timer lifetime race:
its constructor queues an untracked timer task, while `Dispose` cancels and
disposes its CTS without joining that task. Synapse disposes each maintainer
before its tree, matching the package README. The snapshot ZoneTree DLL has
SHA-256 `dc053b4df9f15ce2c2981a0a35b899323d805188797026566c3c7bb7fc0e5db0`.
The exact scheduling was not independently reproduced. This dependency and
bootstrap usage are unchanged; conversion has no ZoneTree calls. Any final
passing test summary must retain this background-exception limitation.

## Limits

General ONNX/Hugging Face language-model import, external ONNX tensors, dynamic
feature axes, state/control-flow execution, graph GPU lowering, adaptive
scheduling and approximate optimization remain unimplemented. CUDA hardware
validation is `not_run_missing_hardware`; this slice changes no GPU kernels
and makes no benchmark claim. Hosted CI and changed-line coverage gates have
not run. The task remains `in_progress`; broader runtime tasks are open.
