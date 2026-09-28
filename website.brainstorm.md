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

## Rejected alternatives

- A generated `gh-pages` branch was rejected because it adds a second mutable
  history and needs force-style publication behavior.
- Rendering the README through Jekyll was rejected because the dedicated site
  is already authored and previewed independently.
- A Node.js or Python static-site generator was rejected by repository policy.
