# Website

`REQ-WEB-001` requires the public Synapse site to be published from its
repository-owned static content at `https://synapse.managed-code.com` whenever
that content changes.

## Acceptance contract

| Acceptance criterion | Evidence |
|---|---|
| `AC-WEB-001-1` relevant `main` changes and manual dispatch build one Pages artifact from `site/` and `docs/images/` | `TEST-WEB-001-1` `PagesWorkflowPublishesCanonicalContent` |
| `AC-WEB-001-2` deployment uses immutable official actions and least-privilege Pages/OIDC permissions | `TEST-WEB-001-1` plus workflow run provenance |
| `AC-WEB-001-3` GitHub Pages, repository About metadata, DNS, TLS, and the live page agree on the custom domain | GitHub API state, DNS answer, successful workflow, and live HTTPS response |

The files under `site/` own the rendered page; `docs/images/` owns its public
diagrams. The workflow copies those sources without modifying them and keeps
documentation links pointed at their canonical repository paths.

## Automatic CI metrics (`REQ-WEB-002`, ADR-010)

Functional test runners publish `test-results.json` from real .NET TRX files;
benchmark aggregation publishes `performance-results.json` from validated raw
JSON. Both use schema version 1. Missing values are null and missing/invalid
evidence is retained. .NET counts and whole-workflow conclusions are separate.
`TASK-WEB-004` keeps Rust formatting, lint and tests independent of .NET/report
failures once the Rust toolchain exists. The .NET step has a shorter timeout
than its job so reporting can run after that step times out. Cancellation or
a job-wide timeout can still prevent publication; missing artifacts remain
explicit, never a passing count. This does not promise unavailable GPU/model
tests execute on hosted hardware.

Pages runs after trusted `main` push/manual workflows complete, downloads the
latest completed verification and performance artifacts by authenticated run
and artifact ID, and publishes `data/latest.json`. The two streams retain
independent commit/run/attempt/time provenance. Failed and partial runs remain
visible. Public CI charts never imply a cross-hardware performance verdict or
replace the dated M2 Pro diagnostics.

| Acceptance criterion | Evidence |
|---|---|
| `AC-WEB-002-1` real nonzero TRX results export immutable JSON; failed, missing, zero and malformed results cannot pass | `TEST-WEB-002-1` `TestResultsPublicationTests` |
| `AC-WEB-002-2` numerical benchmark JSON retains sample counts, nullable metrics, phase scopes and invalid/missing evidence | `TEST-WEB-002-2` `PerformancePublicationTests` |
| `AC-WEB-002-3` trusted completed runs trigger a confined artifact-to-Pages build with independently attributable streams | `TEST-WEB-002-3` `MetricsPipelineTests`, remote Actions evidence |
| `AC-WEB-002-4` the responsive site displays current test counts and scoped benchmark charts, with provenance and honest unavailable states | `TEST-WEB-002-4` rendered desktop/mobile review and public JSON response |
| `AC-WEB-002-5` independent Rust checks run after .NET failures; test-step timeouts leave reporting time without masking failure | `TEST-WEB-002-5` `VerificationKeepsIndependentRustChecksAndReportTimeAfterDotNetFailure`, hosted run evidence |

## Understandable comparisons (`REQ-WEB-003`, ADR-010)

| Acceptance criterion | Evidence |
|---|---|
| `AC-WEB-003-1` charts replace the measurement table; Synapse is highlighted and units/directions are explicit | `TEST-WEB-003-1` shell contract and real desktop/mobile browser review |
| `AC-WEB-003-2` model identities come from raw measurements; legacy enrichment preserves every measured value | `TEST-WEB-003-2` real aggregate and site-data regressions |
| `AC-WEB-003-3` Foundry Local compares with available Synapse data for the selected family/runner/scenario, shows missing series, and separates unlike metric scopes | `TEST-WEB-003-3` rendered selection, missing-model and scope review |
| `AC-WEB-003-4` local Apple GPU charts name their models, sources and limited quality evidence separately from current CI | `TEST-WEB-003-4` rendered local chart review and live Pages revision |
