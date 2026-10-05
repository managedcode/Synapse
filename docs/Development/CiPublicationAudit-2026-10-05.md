# CI publication audit, 2026-10-05

Scope: `TASK-WEB-004`, `REQ-WEB-002`, `AC/TEST-WEB-002-5`, ADR-010.
Existing publication is verified live; this change hardens verification
failure paths. It introduces no new artifact schema or runtime dependency.

## Observed hosted and public state

At source revision `4e86aa252da1b342ac2cfcc1e85766b5b005ad9e`:

- [verify #27](https://github.com/managedcode/Synapse/actions/runs/37157680059)
  succeeded. Actual .NET, JSON export and Rust test steps succeeded on all
  three runners. All three JSON and raw TRX artifacts exist and are unexpired.
- [performance #15](https://github.com/managedcode/Synapse/actions/runs/37157680049)
  succeeded, including the combined report.
- [Pages](https://github.com/managedcode/Synapse/actions/runs/37160170639)
  succeeded. The canonical HTTPS endpoint returns real schema-1
  [latest.json](https://synapse.managed-code.com/data/latest.json), generated
  `2026-10-03T23:32:41.5449156+00:00`, identifying those exact producer runs.

| .NET runner | Total | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| osx-arm64 | 564 | 520 | 0 | 44 |
| linux-x64 | 564 | 519 | 0 | 45 |
| win-x64 | 564 | 519 | 0 | 45 |

These are .NET counts; Rust succeeds separately in the workflow. Discovery
of 564 tests is not proof that every test executed. Unavailable model/hardware
tests remain skipped. Performance has 22/22 expected artifacts, 124 rows,
12 `matched` and 112 explicitly `unreviewed` variants. Short diagnostic samples
do not qualify a statistical performance or broad answer-quality verdict.

The live browser renders the same counts and skips, source SHA/run/attempt,
22/22 artifact status, nullable metrics, quality-unreviewed notes and separate
phase charts. Foundry and MLX comparisons disclose differing weights/device,
memory and timing scope. Dated local M2 Pro charts stay separate from CI.
No site visual changes are part of this repair.

## Repair and checks

Previously a failing .NET test or JSON-export step caused all three subsequent
Rust checks to skip via GitHub's implicit success condition. Each now uses
`!cancelled() && steps.rust-toolchain.outcome == 'success'`. Formatting or lint
failure also cannot suppress the Rust test step. Failures still fail the job.
The .NET step has a 25-minute limit inside a 45-minute job; after normal
preparation, its step timeout leaves time for always-run JSON/TRX reporting.
Cancellation, slow preparation or a job-wide timeout can still prevent reports;
the public missing-artifact path remains explicit. No success is fabricated.

- New real-file workflow contract regression failed before implementation:
  one executed test, one failure, exit 2 (missing 45-minute reporting reserve).
  An initial incorrect filter selected zero tests and failed the minimum policy;
  it is not counted as red behavior evidence.
- `dotnet build Synapse.slnx --configuration Release --no-restore
  -p:UseSharedCompilation=false -m:1 /nr:false`: exit 0, zero warnings/errors.
- `dotnet tests/Synapse.IntegrationTests/bin/Release/net10.0/ManagedCode.Synapse.IntegrationTests.dll
  --treenode-filter '/*/*Website*/*/*' --minimum-expected-tests 25
  --zero-tests-policy strict --report-trx --results-directory
  artifacts/ci-publication-audit/website --coverage --coverage-output
  coverage.cobertura.xml --coverage-output-format cobertura --output Detailed`:
  exit 0, 25/25, zero skips, 50.662 seconds. Existing real-child publication
  regressions cover failed/zero/absent/malformed reports and immutable output.
- `actionlint .github/workflows/verify.yml .github/workflows/pages.yml
  .github/workflows/performance.yml`: exit 0.
- `bash -n .github/scripts/collect-site-results.sh`: exit 0.
- `dotnet format Synapse.slnx --no-restore --verify-no-changes`: exit 0.
- `git diff --check`: exit 0.

Coverage XML and exact local logs are retained under ignored
`artifacts/ci-publication-audit/`; no whole-product coverage claim is made.
The same work session ran Rust locked tests (33 actual tests), fmt/clippy and
NuGet audit successfully; see the adaptive-speculation evidence. The current
full local .NET run is unverified after a reproduced evaluation child timeout;
the old hosted green run does not qualify these unpushed changes.

Manual architecture/security review: only workflow failure conditions and
timeouts change. No permission expansion, new dependency, executed artifacts,
unsafe content injection, schema or data ownership change. Pinned Actions,
run/attempt freshness, hash validation and bounded root-member extraction stay
in place. New hosted completion and public source-revision confirmation are
required before this task is marked verified.

## Delivered executable/workflow revision `501e35d`

[verify #28](https://github.com/managedcode/Synapse/actions/runs/37290201607),
[performance #16](https://github.com/managedcode/Synapse/actions/runs/37290201688)
and the final [Pages deployment](https://github.com/managedcode/Synapse/actions/runs/37293092158)
all succeed. .NET, Rust checks and JSON publication succeed on every runner.

| .NET runner | Total | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| osx-arm64 | 577 | 530 | 0 | 47 |
| linux-x64 | 577 | 529 | 0 | 48 |
| win-x64 | 577 | 529 | 0 | 48 |

The new performance report contains 22/22 expected artifacts, no missing or
invalid artifacts, and 124 rows: 12 matched and 112 unreviewed variants.
The canonical HTTPS JSON equals the uploaded Pages artifact byte-independent
JSON projection exactly. Its generation time is
`2026-10-05T09:56:20.2981639+00:00`; verification and performance both retain
source SHA `501e35dbeab34468205f3e5f34679daa050f9c09`. Every original benchmark
row field/value is unchanged by site publication (optional model enrichment
is excluded from that equality comparison). No missing test artifacts.
The live browser renders these new counts, scopes, run numbers and SHA.

The performance-triggered publication initially kept the still-running
verification stream's previous completed SHA, explicitly shown separately.
Verification completion automatically caused a second publication with both
new streams. Its deploy job queued without an allocated runner or pending
approval, then completed successfully; it was not cancelled or bypassed.
The red/green source regression checks failure conditions; no hosted failure
or timeout branch was deliberately fault-injected. Raw final JSON, producer
artifacts, Pages artifact and watch logs remain in the ignored audit directory.
This post-delivery record changes documentation only.
