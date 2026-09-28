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
