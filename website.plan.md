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
