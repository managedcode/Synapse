# ADR-004: model package sources and import boundary

## Status

Accepted on 2026-09-28 for `TASK-PKG-001..005`. This decision covers source
acquisition and validation; the final compiled Synapse package/chunk schema is
still planned.

## Context

Tests and benchmarks need reproducible small and medium models across several
architectures, but committing binary weights makes the repository and its
history unbounded. Loading arbitrary repository code, floating revisions, or
unverified redirects would also turn model acquisition into a remote-code or
supply-chain boundary.

Synapse needs a format-neutral model identity. GGUF is useful for the first
Qwen2 slice and native baselines, while SafeTensors plus explicit config and
tokenizer files is the first general import source. Neither external container
is the final internal execution format.

## Decision

1. The source-controlled `models/catalog.json` is the reproducibility root. A
   package pins an immutable 40-hex revision, architecture, workload kind,
   source format, precision, parameter count, license, and every required file
   with an exact length and SHA-256 digest.
2. Model payloads are downloaded into an ignored local root. They are not
   committed to Git or Git LFS. CI fetches only the named sets needed by its
   tests.
3. Acquisition accepts HTTPS only. The initial request must match the
   package's source host. Redirects are manual, capped at five, and restricted
   to the source host or trusted Hugging Face storage subdomains. A non-default
   port, downgrade, unrelated host, missing location, or longer chain fails.
4. The downloader checks `Content-Length` when present, enforces the declared
   limit while streaming, computes SHA-256 incrementally, and publishes by an
   atomic move only after length and digest match. Partial files are removed.
5. Package IDs and relative file paths must remain children of the selected
   output root. Leading-dot IDs, rooted paths, `.`/`..` segments, and backslash
   path substitution are rejected before creating a package directory.
6. SafeTensors metadata parsing is bounded by header bytes, tensor count, rank,
   checked shape arithmetic, and non-overlapping payload ranges. Config and
   tokenizer data are inert input; remote/custom model code is never executed.
7. Import dispatches by explicit architecture metadata, not a model brand or
   filename. Unsupported architecture, tensor encoding, tokenizer/template,
   or operator semantics fails before allocation or generation.
8. A later C# compiler converts verified sources into the Synapse manifest,
   typed graph, tokenizer/template, content-addressed chunks, and profiles.
   Conversion remains a no-Python/no-Node repository tool. GGUF stays an
   explicit adapter and benchmark interchange format.

## Consequences

- A Git checkout stays small; fetching a medium model is intentional and
  independently reproducible.
- Catalog maintenance includes upstream license review and exact-file hashes.
- A new download provider needs an explicit trust-policy change and tests.
- Catalog presence means “available reproducibly,” not “executable by
  Synapse.” The support matrix must distinguish package, import, graph, and
  kernel coverage.
- Current cache reuse re-hashes complete files. ZoneTree will later retain
  verified content identities and chunk metadata without weakening startup
  verification or source provenance.

## Verification

- `CatalogRejectsPackagePathTraversal`
- `CatalogRejectsUnsafePackageDirectoryName`
- `OversizedPayloadStopsBeforeWritingBeyondDeclaredSize`
- `RedirectTrustRejectsHttpAndUnrelatedHosts`
- `RedirectTrustAcceptsHuggingFaceStorageHosts`
- real `synapse model fetch` runs for the pinned family and embedding sets
- real SafeTensors header/index checks for SmolLM2, MiniLM, and BGE

The remaining `TASK-PKG-002..005` acceptance tests are not implied complete by
this ADR.
