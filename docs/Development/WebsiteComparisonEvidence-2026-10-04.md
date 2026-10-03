# Website comparison evidence — 2026-10-04

Scope: REQ/AC/TASK/TEST-WEB-003-1..4, ADR-010, and the observed Windows
REQ/TEST-CTX-005-1 corpus regression (ADR-015). Current checkout, no worktree
or stash. No runtime backend, dependency, model weights or benchmark numbers
are replaced by the website work.

## Behavior and raw source

The initial Website suite failed 2/21: the shell contained a table and the
numerical export lacked model identity. The local chart contract also failed
before adding model labels and the 7B weight selector. Legacy publication
regressions exercise real aggregate/site-data child processes, recorded raw
math and exact preservation of each measured field. Changed numerical rows
and unsafe artifact paths cannot obtain a model descriptor.

Authenticated collector data comes from verification run 37151303076 and
performance run 37151302899, source 6ef0213e3070eede6fea808b4746f2108011d5c7.
The artifact IDs and SHA-256 receipts remain under ignored
`artifacts/website-charts/collected/`. Reprojecting raw data enriched all
124/124 existing rows without changing numerical evidence; all 22 expected
performance artifacts were available and model-enrichment issues were empty.
The six Foundry families are not six Synapse decoder implementations.

The public chart selection uses a model family, runner class, scenario/turn
and matching scenario fingerprint. Synapse has a stable first position and
green color. Reported decode/native eval and fresh-process/resident request
wall time remain separate; peak RSS and sampled resident memory also remain
separate. Foundry Local is a local ONNX runtime, not an Azure cloud benchmark.
Unlike weights, threads, cache policies and separate runner jobs are explicit.
Output length, sample count, unreviewed quality and reasoning-only output are
visible. Unsupported/unmeasured Synapse series are not rendered as zero.

The 7B local chart uses both actual alternating rounds from the pinned
2026-09-29 K-quant JSON (Run Y). Median peak RSS / reported speed:
Synapse Q4 4240.9609375 MiB / 17.783821337387096 tokens/s; llama.cpp Q4
4848.703125 MiB / 28.15 native-evaluation tokens/s; Synapse Q8
7231.7265625 MiB / 16.846200565965113 tokens/s; llama.cpp Q8
7987.671875 MiB / 17.495 native-evaluation tokens/s. Different phase
definitions confer no speedup verdict. Exact text parity was checked on
the separate pinned 16-token prompt, not on every long answer. Earlier M2
Pro graphs retain their date/model/source and do not become current CI data.

Windows corpus tests reproduced 3/3 failures using real LF/CRLF documentation
and the real tokenizer before the repair; after canonical LF normalization,
all 6 quality-generator tests passed. The existing failing Windows YAML
assertion now normalizes line endings before inspecting the workflow.
Neither repair confers new model-quality qualification.

## Verification

- Locked restore with `--disable-parallel`: exit 0, six projects. The initial
  sandbox restore was stopped after three minutes without progress; the
  permitted network run completed normally.
- Release solution build with `--no-restore -p:UseSharedCompilation=false
  -m:1 /nr:false`: exit 0, zero warnings/errors.
- Website real-child suite: 24/24 passed, zero skips; isolated copied-binary
  Cobertura run: 24/24, Website complete instrumented files 414/448 lines
  (92.41%). This is not whole-tree or changed-line coverage.
- `dotnet format Synapse.slnx --no-restore --verify-no-changes`: exit 0.
- Rust locked tests: 33 actual tests passed; empty targets do not count as
  required suites. Rust format and clippy `-D warnings`: exit 0.
- NuGet transitive vulnerability audit: exit 0, no reported vulnerabilities
  in six projects. Bash syntax, actionlint and diff checks: exit 0.

The full .NET run requires the existing real subject paths: dotLLM executable
under `_external/dotLLM/src/DotLLM.Cli/bin/Release/net10.0`, version
d88040451d7db56e5dfef9d5754ad0955b0f7fe5; llama-completion under
`/opt/homebrew/bin`, version b29c606e2; model/cache roots under artifacts.
The first full attempt omitted these variables and was stopped after explicit
subject failures; it is not passing evidence. The correctly configured full
run passed 564/564, zero skips, exit 0, 5m 31.482s. Arguments were
`--no-build --no-restore --output Detailed --timeout 35m
--minimum-expected-tests 564 --zero-tests-policy strict --report-trx`;
raw logs/TRX are under `artifacts/website-charts/full-final/` and
`artifacts/chart-full-final-tests.log`. A final metadata-only regression then
caught a false preparation label on an older direct-GGUF fixture. Preparation
is now reported only when that measured subject actually names a `.synapse`
file. Final Release solution build and format verification passed; final
Website regression suite passed 24/24, zero skips, 13.577s. The final copied
coverage run also passed 24/24; its exact count is retained in
`artifacts/website-charts/coverage-final-summary.json`. This final focused
suite follows the metadata-only correction after the complete 564-case run.

Architecture/security review: publication is static C#/Rust-owned tooling
with browser JavaScript. No Node/Python build or benchmark dependency, hidden
runtime engine, generated Git branch, executed artifact content, innerHTML,
or fabricated measurement. Raw extraction remains authenticated, hashed,
bounded and restricted to expected root JSON members; attempt freshness is
rechecked. New functions/types/files stay within limits; site HTML is 398 LOC.
The planned automated architecture/changed-line gates remain unimplemented.

## Rendered browser verification

Real C# Kestrel preview serves source-identical HTML/modules and the genuine
collected publication. CUA browser verification covered CPU smoke/128-token
and dialogue selections, Qwen Synapse/Foundry paired display, Phi's explicit
missing Synapse series, Apple GPU context/KV controls, and 7B Q4/Q8 medians.
At 390 × 844, document and viewport widths are both 390: no horizontal page
overflow, zero tables. A real 404 publication produced the explicit absent
results state with zero live series/test-report cards, not passing counts.
Desktop preview screenshot: ignored `artifacts/website-charts/desktop-preview.jpg`.
Final mobile preview screenshot is retained in
`artifacts/website-charts/mobile-preview.jpg`. Zero browser console errors
were observed. Live revision, screenshots and delivery follow after the push;
this committed evidence snapshot does not claim a deployment before it runs.
