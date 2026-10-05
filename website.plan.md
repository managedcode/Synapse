# Synapse public website plan

## `TASK-WEB-001`: GitHub Pages delivery

- [x] Select the hand-authored `site/` files and public SVGs as the canonical
  source.
- [x] Record the GitHub Pages topology in ADR-010 and define Website
  requirements and acceptance evidence.
- [x] Add a failing repository contract test for source triggers, immutable
  Pages actions, deployment permissions, the custom domain, and public assets.
- [ ] Add the GitHub Pages workflow around the parallel site, then pass the
  focused test.
- [ ] Validate the staged static artifact and rendered desktop/mobile site.
- [ ] Commit only Website delivery files plus the explicitly identified site
  content, push `main`, and observe the real Pages deployment to completion.
- [ ] Configure and verify the Pages custom domain, HTTPS, GitHub About URL,
  repository topics, and live canonical page.

Completion requires the remote workflow to finish successfully and the custom
HTTPS URL to serve the published revision. A local render or pushed commit is
not delivery evidence.

## `TASK-WEB-002`: automatic CI results and metrics

- [x] Inspect KeyLoad's JSON artifact, aggregation and Pages publication flow.
- [x] Extend ADR-010 with schemas, independent provenance and the artifact
  trust boundary; register REQ/AC/TASK/TEST-WEB-002.
- [x] Add failing real-process regressions for TRX-to-JSON, benchmark JSON,
  invalid/partial evidence and immutable publication.
- [x] Export per-runner test JSON and aggregated numerical benchmark JSON.
- [x] Collect trusted completed `main` artifacts and build public
  `data/latest.json`; trigger Pages after either source workflow finishes.
- [ ] Render current test counts and scoped/filterable benchmark metrics,
  provenance, nullable values and missing evidence on desktop/mobile.
- [x] Run format/build/analyzers, focused regressions, security and
  architecture checks, and focused coverage; record exact evidence.
  Local 19/19 Website, 2/2 hosted compatibility and 32 Rust tests pass;
  Website source coverage is 92.64%. See
  `docs/Development/WebsiteMetricsEvidence-2026-10-03.md`.
- [ ] Commit scoped changes, push and observe JSON publication and Pages
  deployment; verify the canonical HTTPS metrics endpoint.

Keep the task `in_progress` until remote publication and live evidence exist.
This does not close the statistical benchmark or broader runtime plans.

## `TASK-WEB-004`: failure-independent verification and live audit

- [x] Confirm hosted .NET/Rust steps, JSON artifacts, Pages and the public
  endpoint for `4e86aa2`; preserve actual skips and diagnostic quality states.
- [x] Red contract regression `TEST-WEB-002-5` on the existing workflow.
- [x] Give the .NET test step a reporting reserve and keep Rust checks
  independent of earlier failures after toolchain installation.
- [x] Focused Website 25/25, format/build/workflow lint and evidence review;
  `docs/Development/CiPublicationAudit-2026-10-05.md` records exact scope.
- [x] Deliver the workflow at `501e35d` and confirm hosted verification,
  performance JSON and Pages; the canonical endpoint and browser display the
  exact new artifact, including 577 tests per OS and 124 benchmark rows.
  Failure conditions have a red/green source contract; hosted failure/timeout
  branches were not deliberately fault-injected. Broader qualification stays open.
