# Synapse website delivery decisions

Date: 2026-09-28. Status: accepted for `TASK-WEB-001`.

## Decisions

1. Publish the hand-authored static website under `site/` without rewriting or
   regenerating its HTML, CSS, or JavaScript.
2. Publish through the repository's GitHub Pages environment from a custom
   GitHub Actions workflow. The workflow builds only when the README, its
   public images, the site shell, or the workflow itself changes.
3. Use GitHub's official Pages actions pinned to immutable revisions. The
   repository adds no static-site generator and no Node.js or Python build
   dependency.
4. Stage the files under `site/` and the public SVGs under `docs/images/` into
   one immutable Pages artifact during CI.
5. Configure `synapse.managed-code.com` through the GitHub Pages API and keep
   the repository About website URL aligned with the published canonical URL.

## 2026-10-03 automatic CI metrics decisions

- `REQ-WEB-002`: publish real functional-test and benchmark JSON after each CI
  run, then rebuild Pages automatically from those artifacts.
- Follow KeyLoad's artifact-to-site delivery pattern using C# and shell;
  its Node tooling is outside Synapse's permitted dependencies.
- Keep latest verification and performance runs independently attributable;
  display incomplete and failed runs, never substitute stale passing numbers.
- Add a dedicated current CI section. Existing local M2 Pro charts are dated
  evidence and cannot be overwritten with measurements from hosted machines.
- Validate schemas, runner labels, artifact/run/attempt identity and extraction
  paths. Treat every downloaded file as data, never executable build input.

## Original rejected alternatives

2026-10-05 (`TASK-WEB-004`): live audit confirms the artifact-to-Pages flow
already works. Harden failure paths: earlier .NET/report failures must not
skip independent Rust checks, and a bounded test step must leave reporting
time. Keep genuine skips, independent revisions and diagnostic quality states;
do not relabel all discovered tests or all benchmark rows as qualified passes.

- A generated `gh-pages` branch was rejected because it adds a second mutable
  history and needs force-style publication behavior.
- Rendering the README through Jekyll was rejected because the dedicated site
  is already authored and previewed independently.
- A Node.js or Python static-site generator was rejected by repository policy.
