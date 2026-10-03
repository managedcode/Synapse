# Website metrics publication evidence — 2026-10-03

Scope: `REQ-WEB-002`, `AC-WEB-002-1..4`, `TASK-WEB-002`,
`TEST-WEB-002-1..4`, ADR-010. Use the current checkout; preserve pre-existing
model/runtime/docs changes. No worktree, stash, dependency or model change.

## Behavior-first verification

- Benchmark JSON export: initial 4/4 failed because `aggregate --json` was
  unsupported (exit 2); final 4/4 passed. Existing hosted-report 2/2 passed.
- TRX export: initial five real-child tests failed because `test-report` was
  unsupported; the final seven test cases passed. They execute real TUnit
  fixtures and mutate real XML for invalid input cases; no service double.
- Pipeline/site data: the initial three cases had two failures (no trigger
  and no site-data command). Four additional source/report regressions exposed
  invalid-data exception handling and scalar metadata acceptance before the
  repairs. The final positive case executes real TUnit -> TRX -> test JSON and
  recorded raw benchmark -> numerical JSON -> public site JSON.
- Final Website suite: 19/19 passed, zero failed/skipped, with the real TRX
  retained under ignored `artifacts/website-publication/final-tests/`.

## Commands and outcomes

| Command | Outcome |
|---|---|
| `dotnet restore Synapse.slnx --locked-mode` | Exit 0, all six projects |
| `dotnet build Synapse.slnx --configuration Release --no-restore --disable-build-servers -m:1` | Exit 0, zero warnings/errors |
| `dotnet format Synapse.slnx --no-restore --verify-no-changes` | Exit 0 after the final constructor style fix |
| `dotnet artifacts/website-publication/coverage-snapshot/ManagedCode.Synapse.IntegrationTests.dll --treenode-filter '/*/ManagedCode.Synapse.IntegrationTests.Features.Website/*/*' --minimum-expected-tests 19 --coverage --coverage-output-format cobertura --coverage-output /Users/ksemenenko/Developer/Synapse/artifacts/website-publication/coverage.cobertura.xml --results-directory /Users/ksemenenko/Developer/Synapse/artifacts/website-publication/final-tests --report-trx` | Exit 0, 19 passed, 17.105s |
| `dotnet experiments/Synapse.ReferenceBenchmarks/bin/Release/net10.0/Synapse.ReferenceBenchmarks.dll test-report --results artifacts/website-publication/final-tests --runner osx-arm64 --output artifacts/website-publication/final-test-results.json` | Exit 0; actual 19 passed, zero failed/skipped |
| `cargo test --manifest-path native/Cargo.toml --locked` | Exit 0; 32 actual tests, empty unit/doc targets do not count as required suites |
| `cargo fmt --manifest-path native/Cargo.toml --all --check` | Exit 0 |
| `cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings` | Exit 0 |
| `dotnet list Synapse.slnx package --vulnerable --include-transitive` | Exit 0; no reported vulnerable packages in six projects |
| `actionlint .github/workflows/pages.yml .github/workflows/verify.yml .github/workflows/performance.yml` | Exit 0 |
| `bash -n .github/scripts/collect-site-results.sh` | Exit 0 |
| `git diff --check`, `git diff --cached --check` | Exit 0 |

Rust commands use `PATH=/opt/homebrew/opt/rustup/bin:$PATH`. Sandbox feed/IPC
access was granted for required checks. Coverage runs from a copied binary
directory to preserve the normal build. C# XML analysis deduplicates filename
and line number: Website source coverage is 340/367 (92.64%). Overall benchmark
assembly coverage is 9.3% for this focused filter; this is not a full-solution
or changed-line coverage claim.

## Artifact and source qualification

The real collector verified existing GitHub artifact SHA-256 values and
retained independent metadata for verification run `36547564346` (failure)
and performance run `36547564428` (success), both source
`7a219c6cb6fdb851962a3f23a2f141abde20679d`. These older runs have no new JSON,
so the produced `current-latest-validated.json` truthfully contains absent
test artifacts and a null benchmark report. Historical Markdown-only
artifacts are not relabeled as numerical evidence.

An independent source export consists of `git archive HEAD` plus exactly the
staged publication patch; it contains none of the pre-existing model changes.
Its first build caught an incorrectly positioned staged switch insertion;
the staged-only patch coordinates were corrected before publication. The
corrected export passed locked restore, Release solution build (5.80s,
zero warnings/errors), 19/19 Website tests (12.268s) and 2/2 hosted-report
compatibility tests (263ms), all exit 0. Remote publication follows separately.

Architecture review: no new runtime engines/dependencies, generated branches,
or Node/Python tooling. Files/types stay within repository size limits. Only
trusted completed `main` push/manual workflow runs may publish; collector
checks attempt freshness, artifact digests, bounded content and expected root
members without extracting untrusted paths. The site uses textContent for
report data and validates GitHub links and numeric/schema fields.

## Current limitations

The C# preview serves source-identical HTML/JS with HTTP 200 and a truthful
404 before metrics data exists. CUA exposes no enabled browser/app surface, so
desktop/mobile rendering and browser JavaScript execution are unverified.
The broader runtime/statistical benchmark plans remain open; these tests do
not establish a performance winner. `TASK-WEB-002` stays `in_progress` until
remote/live evidence and the outstanding browser review are recorded.

Automatic approval review initially rejected the scoped commit/push to `main`
because the dependency-repair authorization does not cover this feature. No
commit or push occurred during that attempt. On 2026-10-03 the user explicitly
authorized pushing these prepared changes to `main`; publication proceeds
under that authorization.

## Workflow name regression after publication

Commit `00567dbba63f217ba8387829bb67131c877b5731` reached `origin/main` and
created Pages run `37128235031`, verification run `37128235060`, and performance
run `37128235064`. Run discovery exposed the actual performance workflow name
`performance-diagnostic`, while the Pages subscription used `performance`.
The improved source contract derives each upstream name from its real workflow
file. It failed first (1/1, exit 2) before correcting the subscription.
These run IDs establish dispatch, not completed test or deployment evidence.
After the correction, Release test-project build passed with zero warnings
or errors, the 19-case Website suite passed (exit 0, 13.971s), scoped format
verification passed, and actionlint/diff checks passed. Sandbox IPC-denied
attempts were rerun with required permissions and are not counted as passes.
