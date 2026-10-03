# ADR-010: GitHub Pages website delivery

Status: Accepted. Date: 2026-09-28.

## Decision

`TASK-WEB-001` publishes the public project website from `main` with a custom
GitHub Actions workflow and the `github-pages` deployment environment. The
workflow takes the hand-authored HTML, CSS, JavaScript, and favicon under
`site/`, combines them with the public SVGs under `docs/images/`, uploads one
static Pages artifact, and deploys it with GitHub's Pages action.

The workflow runs on relevant pushes and manual dispatch. Its permissions are
least-privilege: read repository contents, write Pages, and mint the Pages OIDC
token. All third-party workflow references are official GitHub actions pinned
to immutable commit SHAs. Deployments are serialized without cancelling an
in-progress publication.

The canonical public origin is `https://synapse.managed-code.com`. GitHub owns
the Pages custom-domain and HTTPS settings; DNS remains externally owned. The
repository About website URL points at the same canonical origin.

## Consequences

Site or public-image changes automatically republish the artifact without a
generated branch or build-time content rewrite. GitHub Pages is a delivery
platform, not a Synapse runtime, test, import, conversion, or benchmark
dependency. Local inference continues to require neither GitHub nor a network.

The repository does not maintain a generated `gh-pages` branch. A successful
local source check or push is insufficient evidence: publication is complete
only after the remote Pages workflow succeeds and the canonical HTTPS URL
serves that revision.

## 2026-10-03: CI result publication (`TASK-WEB-002`)

Completed trusted `main` runs of `verify` and `performance` also trigger the
Pages workflow. Pull-request and fork runs cannot publish. The build checks
out trusted `main` source, reads Actions artifacts as data only, and selects
the latest completed push/manual run of each workflow. Both streams retain
their own run ID, attempt, source SHA, conclusion and time; they may describe
different commits. Serialized deployments select current runs when building,
so an older queued completion cannot deliberately republish its old snapshot.

Each verification runner exports a version-1 JSON projection of real .NET TRX
results. Zero tests, absent reports, malformed reports and failing tests never
become a passing result. Rust remains part of the workflow conclusion; .NET
counts explicitly name their scope. Performance aggregation exports a
version-1 JSON projection of the validated raw benchmark artifacts, including
missing/invalid evidence, nullable metrics and measured sample counts. Native
eval and reported decode rates, and request and fresh-process wall times,
retain distinct scopes. These diagnostics confer no statistical winner verdict.

A C# command assembles `data/latest.json` with independent run provenance,
validated test reports and performance rows. The public site displays these
CI results separately from dated local Apple M2 Pro measurements. JSON and
upstream run links are public; raw artifacts remain on GitHub Actions. Missing
or expired artifacts are shown explicitly. No generated Git commits, runtime
backend changes, Node.js or Python tooling are introduced.

The collector uses authenticated API artifact IDs from the selected run and
attempt, expected names only, bounded downloads and a confined extraction
directory. Artifact paths, schemas and runner identities are checked before
publication. Only deployment receives Pages/OIDC write privileges.

## 2026-10-03: understandable comparison charts (`TASK-WEB-003`)

Replace the public row table with model/runner/scenario comparison charts.
Version-1 rows gain an optional model descriptor: family, file/catalog label,
weight fingerprint, execution policy, scenario fingerprint and runner label.
Absent descriptors stay explicitly unknown. Pages also collects the expected
raw JSON members from authenticated bounded artifacts. Legacy summaries may
be enriched only when reprojecting the validated raw file reproduces every
existing row field exactly; enrichment cannot change measured values.

Synapse has a fixed visible position and color. Foundry Local comparisons use
the same model family, runner class, scenario and turn, with separate weights,
threads and cache policies disclosed. Runner classes do not prove identical
physical hosts. Native evaluation rates and reported decode rates, resident
request and fresh-process wall time, and resident and peak RSS remain separate
charts. Missing Synapse measurements stay visible; no speedup or quality
verdict is inferred from unlike formats or unreviewed answers. Dated local
Apple GPU charts remain clearly separate from current hosted CI.
