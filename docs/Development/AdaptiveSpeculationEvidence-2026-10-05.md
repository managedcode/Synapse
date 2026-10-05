# Adaptive speculative depth evidence, 2026-10-05

Scope: `REQ-SPC-005`, `AC-SPC-005-1..3`, `TASK-SPC-005`,
`TEST-SPC-005-1..3`; amendment to ADR-020. Identifier 004 remains reserved
for planned self-speculation in the FlyBrain plan. Experimental local implementation;
the task remains in progress pending paired performance qualification.

## Behavior and boundaries

`SpeculativeDecoding.Generate(..., draftTokens: 3, adaptiveDepth: true)` uses
the draft count as a cap; CLI `--draft-tokens auto` caps it at three. Managed
CPU supports adaptive caps up to seven; adaptive GPU caps above three fail
explicitly because wider verification did not preserve real-model parity. Fixed
counts and their default three remain available. The controller compares
EWMA committed tokens divided by draft-plus-target round wall time. Two
initial samples per depth include zero (ordinary target decode), a 5% margin
keeps small apparent gains on the plain path, and rotating probes back off
from 16 to 128 exploitation rounds. Actual per-depth counters and timings
include clipped tails; tails do not bias the estimator.

This independently applies the measured-cost idea in
[TensorFold bcb8f01](https://github.com/ashhart/TensorFold/commit/bcb8f01).
It uses Synapse's existing external draft and target verification. No MTP
checkpoint, GPU startup calibration or confidence-based per-level stopping
is implemented. The upstream performance claim is not a Synapse result.

The initial contract-red build produced eleven compiler errors for the
missing controller/API/measurements. Final regressions execute actual
managed/Metal tiny-model math and CLI child processes; controller tests use
its real numerical estimator directly, without service doubles. Tests cover
agreeing/disagreeing drafts, ordinary/speculative transitions, requests of
one/two tokens, valid accounting, cost drift, a 5% margin, clipped learning
and explicit invalid-cost failures. Final adaptive assertions use
`SequenceEqual`, CPU cap seven and Metal cap three. A prepared Qwen2.5 0.5B
Metal/f16-KV regression compares 128 real tokens against ordinary decode;
another requires wider adaptive GPU windows to fail explicitly. That guard
test failed before implementation because cap four did not throw.

## Executed checks

Host: macOS ARM64; .NET SDK 10.0.401; pinned Rust 1.98.1 resolved through
`/opt/homebrew/bin/rustup which cargo`. The sandbox prevents test/format IPC
socket creation and hides Metal, so those commands use narrow escalations.
The sandboxed test attempt failed before running tests; it is not a pass.

- `dotnet restore Synapse.slnx --locked-mode`: exit 0, all up-to-date.
- `dotnet build Synapse.slnx --configuration Release --no-restore
  -p:UseSharedCompilation=false -m:1 /nr:false`: exit 0, zero warnings/errors.
- `dotnet format Synapse.slnx --no-restore --verify-no-changes`: exit 0.
- `dotnet list Synapse.slnx package --vulnerable --include-transitive
  --no-restore`: exit 0, no vulnerable packages from configured NuGet source.
- `cargo test --manifest-path native/Cargo.toml --locked`: exit 0,
  33 actual tests (1 + 2 + 23 + 7); empty binary/doc-test targets are not
  counted as suites proving behavior.
- `cargo fmt --manifest-path native/Cargo.toml --all --check`: exit 0.
- `cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets
  -- -D warnings`: exit 0.
- `dotnet tests/Synapse.IntegrationTests/bin/Release/net10.0/ManagedCode.Synapse.IntegrationTests.dll
  --treenode-filter '/*/*Speculation*/*/*' --minimum-expected-tests 18
  --zero-tests-policy strict --report-trx --results-directory
  artifacts/adaptive-speculation/bounded-final --coverage
  --coverage-output coverage.cobertura.xml --coverage-output-format cobertura
  --output Detailed`: exit 0, 18/18, zero skips, 1m 27.781s, including real Metal.
- `git diff --check`: exit 0.

Coverage is isolated full-file coverage from those focused tests, not
whole-product or changed-line qualification: `AdaptiveDepth.cs` 53/53,
`SpeculationLoop.cs` 75/75, `SpeculativeDecoding.cs` 33/34 and CLI
`SpeculationArguments.cs` 58/59 executable lines. Unchanged token identity
is 4/15 in this suite. Raw XML/TRX and the C# aggregation program are under
ignored `artifacts/adaptive-speculation/`.

Manual architecture/security review: bounded eight-depth request-local
state, no new dependency, file format, persistence, network or execution
boundary. Direct-slot ownership remains under both model gates. Plain decode
overwrites its next position; draft catch-up overwrites rejected tails before
reuse. All committed tokens still come from target logits. Timing diagnostics
are finite and aggregate counts account for every reported target pass. New
files/types/functions remain within repository size limits. Planned automated
architecture/security gates are not represented as executed checks.

## Full suite and diagnostic pilot

An initial full-suite attempt used a nonexistent checkout baseline path,
failed the three native smoke prerequisites and was canceled. The verified
installed `/opt/homebrew/bin/llama-completion --version` reports build 10964,
commit `b29c606e2`; the corrected run uses that executable and its actual version.
The checked-out dotLLM source is independently verified as
`d88040451d7db56e5dfef9d5754ad0955b0f7fe5`. A two-second sample and managed
stack report show active CPU attention in the existing
`KvPageActivationTests.KvPagesCoveringThePrefixEqualDense` regression, rather
than an adaptive-depth stall. Stack reports are retained in the local logs.

The corrected full command was `dotnet test Synapse.slnx --configuration
Release --no-build -- --minimum-expected-tests 1 --zero-tests-policy strict
--report-trx --results-directory artifacts/adaptive-speculation/full-final
--progress on --output Detailed`, with verified subject executable/version
environment variables. It was stopped at 35 minutes, exit 130, after
`OptimizationEvaluationTests.EvaluationRotatesPairsAndPreservesIdenticalInputAndScoreTraces`
hit its child's two-minute timeout. The isolated real test reproduced that
timeout (one failure, exit 2, 2m 05s). Its path has no speculation flags and
was not changed here; this observation does not establish the root cause.
The full .NET gate is neither complete nor passing.

The pilot uses one prepared Qwen2.5 0.5B Q8_0 model on Apple M2 Pro Metal,
f16 KV, eight threads, a five-token prompt and 128 generated tokens. The
draft is the same model with one dropped layer. Each mode has a warm-up
and three measured fresh processes, with alternating order. Raw JSON/stderr
and the C# reporter remain under `artifacts/adaptive-speculation/`.

The first cap-seven pilot is rejected: all three adaptive outputs differed
from dense at token position 36 (dense 279, adaptive 30083). Fixed depth three
matched dense. Wider windows enter a different native GEMM path; no native
kernel change or numerical parity claim is made. Rejected raw files remain
under `rejected-wide-pilot/` with `rejected-wide-pilot-summary.json`.

After bounding GPU auto to three, all 128 tokens match dense, fixed and auto
in all three measured pairs. Each adaptive process recorded rounds at depth
zero: 115, one: 3, two: 3, three: 2. Reporter exit 0.

| Mode | Median generation wall ms | Median reported decode tokens/s |
|---|---:|---:|
| Dense | 892.034 | 154.630 |
| Fixed depth 3 | 1901.460 | 69.252 |
| Auto cap 3 | 1532.829 | 86.751 |

Dense remains fastest. Auto reduces fixed-draft cost but sampling/prefill and
catch-up still cost more than dense. These three pairs are diagnostic, not a
statistical or broad quality/speed qualification. Exact token parity is not
a semantic answer-quality verdict.

Final pilot SHA-256 identities:

- CLI: `feb90b481a5c2abf037b819381f0ea63d66cc217b4beb86f687f779c2569d209`.
- Runtime: `6969fcb9670489e935cc935cdc6202e61932bd0415bbdf6bcd963f37f60d78be`.
- Prepared model: `b67620a926cd299931b0a7cbfbd31f112c893832658cefa64e67f4d4a21f514d`.

## Limitations

Initial sampling, draft prefill and catch-up after plain rounds can cost more
than speculation saves. No speed improvement is guaranteed. No NVIDIA run,
MTP-family support, broad corpus qualification, statistical paired speed
claim, commit/push or deployment is established by this checkpoint.

## Additional 7B diagnostic and hosted delivery

Executable revision `501e35d` is pushed; full hosted verification and public
JSON publication succeed (see `CiPublicationAudit-2026-10-05.md`). Hosted
Metal skips are real: Apple's paravirtual device reports GPU family zero.
Local real Metal coverage above is separate from that hosted evidence.

An additional existing Qwen2.5-7B-Instruct-1M Q8_0 source was losslessly
prepared with the C# compiler, source SHA-256
`d81ca8ca04442ed5ff7d036fa546183170b8269281380bed977431a5449c84b2` and package
identity `9c1e2aca1f23bcc51f06802cac5e9e25e2835e0f9a6988faf92971a2c3986a16`
(the identity is not a whole-file hash). External draft: prepared 0.5B Q8_0,
without layer drop. Metal/f16 KV, eight threads, context 512, the same
37-token chat prompt as the earlier 7B diagnostic, and 256 generated tokens.
One warm-up per mode and three alternating measured fresh-process pairs.
Final compiler/helper/12 generation processes and the C# reporter exit 0;
every fixed and auto output has all 256 tokens in the same order as its paired
dense output. Raw prompt, tokens, JSON/stderr, C# reporter, orchestration,
summary and fuller limits are under `artifacts/adaptive-speculation/seven-billion/`.

| Mode | Median generation wall ms | Median reported decode tokens/s |
|---|---:|---:|
| Dense | 13085.350 | 20.313 |
| Fixed 3 | 10466.081 | 26.105 |
| Auto cap 3 | 12371.546 | 24.239 |

Fixed 3 remains fastest here, and remains the default. Adaptive choices vary
by round, including zero in the third. Other user-owned CPU work exists on
the shared host; identical load/thermal conditions are not established.
Only three pairs on one prompt/model pair, with no semantic answer-quality
or statistical speedup verdict. This does not close `TASK-SPC-005` or add
these local diagnostic rows to the public hosted benchmark stream.
