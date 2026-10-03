# Pipeline model parameters and ignored benchmark output — 2026-10-03

Scope: `REQ-BMK-001`, `AC-BMK-001-5..6`, `TASK-BMK-001`,
`TEST-BMK-001-5..6`, ADR-005. Preserve pre-existing staged model conversion,
compiled-package, quantization and runtime work in the current checkout.

Manual `performance-diagnostic` runs expose the GGUF catalog ID/file, MLX
catalog ID, Foundry model-set file and the smoke prompt/token/continuation
qualification. Manual `verify` runs expose the GGUF selection, functional
model sets and Foundry anchor. Repository variables with the same `SYNAPSE_*`
environment names select defaults for automatic runs; a manual input takes
precedence, then a repository variable, then the workflow's configured fallback.
The smoke expected continuation remains a quality gate for the selected weights.
Local fixture defaults read the versioned catalog/model-set data; C# fixture
lookup does not contain fallback model IDs or assumed source filenames.

`/benchmarks/` is ignored. Required model-set/scenario definitions are retained
under `experiments/Synapse.ReferenceBenchmarks/Features/Benchmarking/`, and
recorded raw report regression fixtures under
`tests/Synapse.IntegrationTests/Features/Benchmarking/Fixtures/`.
Existing ignored benchmark output remains on disk. Migrated evidence keeps its
original bytes and date; it does not represent a new measurement.

Historical README/site evidence links pin the pre-removal commit `d679cb7`
so its dated raw results stay reachable. The current benchmark footer links
to the tracked feature documentation. Smoke report labels use the declared
raw token limit, and MLX labels no longer assume Qwen weights. The parser
regression declaring sixteen tokens preserves its original eight observed
tokens and marks quality mismatch; it is not newly measured evidence.

## Red/green evidence

- Workflow contracts: four intended missing-input failures, then 4/4 green.
- Selected model fixtures: two failures before environment propagation, then
  2/2 green using two genuine tiny GGUF files and a real Foundry planning process.
- Tracked inputs: the clean-input contract and a hosted reporter failed on
  missing files before the thirteen byte-identical inputs were copied.
- Configurable report labels: two intended failures before the reader repair,
  then 6/6 publication and 2/2 hosted compatibility cases green.

## Exact verification commands and outcomes

Independent source export: `/private/tmp/synapse-pipeline-scope-review`,
`git archive HEAD` plus only these changes, with no `benchmarks/` directory.
It excludes the separately staged compiler/conversion/runtime work. The
working checkout's already staged preparation steps consume the same model
parameters; those steps remain attached to their separate compiler delivery.
No worktree or stash was used. Evidence outputs below are relative to the export.

| Command | Exit and evidence |
|---|---|
| `dotnet restore Synapse.slnx --locked-mode` | 0, six projects |
| `dotnet build Synapse.slnx --configuration Release --no-restore --disable-build-servers -m:1` | 0, zero warnings/errors |
| `dotnet format Synapse.slnx --no-restore --verify-no-changes` | 0 after final style repairs |
| `dotnet artifacts/parameter-check/coverage/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.Website/*/*' --minimum-expected-tests 21 --coverage --coverage-output-format cobertura --coverage-output artifacts/parameter-check/coverage.cobertura.xml --results-directory artifacts/parameter-check/final-website --report-trx` | 0, 21 passed, zero skipped, 20.219s |
| `dotnet tests/Synapse.IntegrationTests/bin/Release/net10.0/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.Benchmarking/<suite>/*' --minimum-expected-tests 1 --results-directory artifacts/parameter-check/<suite> --report-trx` | 0 for all eight suites below, 22 total passed, zero skipped |
| `dotnet list Synapse.slnx package --vulnerable --include-transitive --no-restore` | 0, no reported vulnerabilities in six projects |
| `actionlint .github/workflows/performance.yml .github/workflows/verify.yml .github/workflows/pages.yml` | 0 |
| `bash -n .github/scripts/collect-site-results.sh` | 0 |
| `git diff --check`, `git diff --cached --check` | 0 |
| `cargo test --manifest-path native/Cargo.toml --locked` | 0, 32 actual tests; empty unit/doc targets are not counted |
| `cargo fmt --manifest-path native/Cargo.toml --all --check` | 0 |
| `cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings` | 0 |

The eight suites and counts are `WorkflowModelParameterTests` (4),
`ConfiguredFixtureSelectionTests` (2), `BenchmarkingInputContractTests` (1),
`DiagnosticMatrixReportTests` (2), `DialogueReportTests` (2),
`HostedArtifactReportTests` (2), `FoundryLocalPlanTests` (6), and
`FoundryLocalRunTests` (3). The latter streamed through the actual cached
Foundry anchor with `SYNAPSE_FOUNDRY_CACHE=/Users/ksemenenko/Developer/Synapse/artifacts/foundry-local`;
the live/uncached/recorded suite took 10.253s. No model service double was used.
After removing the remaining C# fallback IDs, the selected-source tests (2)
and real Foundry tests (3) passed again, and `SharedQwenFixtureMatchesPinnedDigest`
passed against the actual default catalog package (1; 750ms). The scoped
validation therefore covers 44 distinct .NET cases, with the affected cases
rerun after the fixture refactor. The final Foundry run took 13.842s.
Rust commands ran in the working checkout using
`PATH=/opt/homebrew/opt/rustup/bin:$PATH`. Scoped runtime sources have no Rust changes.

Coverage ran from copied binaries so regular outputs stayed intact. Overall
ReferenceBenchmarks line coverage is 10.5% for this focused Website run,
not full-solution or changed-line coverage. The complete .NET runtime suite
and browser visual execution were not rerun for this configuration/report slice.
Deduplicated line coverage for the changed `HostedEvidenceReader.cs` is
93/98 (94.90%), computed from the actual Cobertura output with C# XML parsing.

## Delivery boundary

The user explicitly authorized pushing the publication changes to `main`.
The parameter/configuration correction follows that authorization, with
unrelated staged work preserved. At the pre-push check, local and remote
`main` both point to `d679cb7`. Verification `37128480434`, performance
`37128480466` and Pages `37128483767` remain queued/pending; no live updated
JSON is claimed. `TASK-BMK-001` and the live website delivery gate remain open.
