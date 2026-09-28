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
