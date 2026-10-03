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

Pages runs after trusted `main` push/manual workflows complete, downloads the
latest completed verification and performance artifacts by authenticated run
and artifact ID, and publishes `data/latest.json`. The two streams retain
independent commit/run/attempt/time provenance. Failed and partial runs remain
visible. Public CI tables never imply a cross-hardware performance verdict or
replace the dated M2 Pro diagnostics.

| Acceptance criterion | Evidence |
|---|---|
| `AC-WEB-002-1` real nonzero TRX results export immutable JSON; failed, missing, zero and malformed results cannot pass | `TEST-WEB-002-1` `TestResultsPublicationTests` |
| `AC-WEB-002-2` numerical benchmark JSON retains sample counts, nullable metrics, phase scopes and invalid/missing evidence | `TEST-WEB-002-2` `PerformancePublicationTests` |
| `AC-WEB-002-3` trusted completed runs trigger a confined artifact-to-Pages build with independently attributable streams | `TEST-WEB-002-3` `MetricsPipelineTests`, remote Actions evidence |
| `AC-WEB-002-4` the responsive site displays current test counts and filterable scoped benchmark rows, with provenance and honest unavailable states | `TEST-WEB-002-4` rendered desktop/mobile review and public JSON response |
