# Benchmarking

The benchmark system launches Synapse, dotLLM, LLamaSharp, and direct llama.cpp as isolated
processes against a shared pinned GGUF and input-token manifest. A result is
eligible only when model family/source, tokenizer, template, prompt tokens,
context/output limits, sampling, hardware, thread count, power mode, and cache
state are equivalent or explicitly classified.

Initial audited upstream revisions on 2026-09-27:

| Subject | Revision/version | Role | License boundary |
|---|---|---|---|
| dotLLM | `d88040451d7db56e5dfef9d5754ad0955b0f7fe5` | Pure-.NET reference and required competitor | GPL-3.0, separate process/source checkout |
| LLamaSharp | `abf614ef899464aced879b19aa5dca53f15e5525` / NuGet 0.27.0 observed | llama.cpp-backed required competitor | MIT wrapper plus native llama.cpp artifacts |
| ZoneTree | `13ee11e19007301fdea72b9210de62f6257f4929` / NuGet 1.9.8 | Embedded durable metadata/index store | MIT runtime dependency |

The first diagnostic Qwen2.5 result now has raw samples in
`benchmarks/results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-smoke.json`. It uses
three warm-ups plus five measured, rotated, cold-process samples. That is useful
engineering evidence, but below the release contract of 30 paired samples and
therefore carries no formal winner verdict.

Every measured row records load, TTFT, total generation, decode tokens/s,
subject wall time, process CPU time, average CPU cores, and observed working
set. Energy/power is a separate nullable status. `not_run_missing_privilege`
or `not_run_missing_hardware` never acquires a numeric zero.

The process memory envelope is collected before model load through completion
for every reference subject. Peak resident working set includes managed and
native pages; macOS physical footprint is recorded separately via
`proc_pid_rusage`. Virtual/private address-space values are **not** interpreted
as resident memory (private virtual size is unavailable from the current
macOS process API and remains null). Synapse CLI and LLamaSharp also emit live
CLR heap and cumulative managed allocation diagnostics. These cannot be subtracted from
RSS to obtain native bytes. The sampler records its count; unavailable metrics
remain null. The existing dated smoke samples predate this schema and are not
retroactively relabeled.

Profiling is a separate diagnostic run, not part of paired timing samples:
open-source `dotnet/diagnostics` supplies CLR counters/traces/heap inspection,
`samply` can inspect native CPU stacks on macOS and Linux, and installed Xcode
Instruments supplies Mac Allocations/Leaks/Metal traces. Linux-only heaptrack
is not a Mac collector. Profiling overhead and native allocator attribution
must be reported with the trace, never silently merged into baseline timing.

The C# `matrix` command in `experiments/Synapse.ReferenceBenchmarks` now runs
Synapse, dotLLM, LLamaSharp, and direct llama.cpp in rotated fresh-process
rounds. It requires explicit executable paths, source/model token IDs, a
locked expected continuation, thread/output limits, and a new output path;
defaults are three warm-ups and five measured rounds. The output is immutable
raw JSON with binary/model SHA-256 digests, every subject's original JSON,
quality status, OS memory peaks, and sample counts. A mismatch is retained as
`ineligible_quality_mismatch`, not erased or awarded a throughput verdict.
Run `dotnet experiments/Synapse.ReferenceBenchmarks/bin/Release/net10.0/Synapse.ReferenceBenchmarks.dll matrix`
without arguments for the exact option list. This diagnostic runner is not yet
the 30-pair randomized release benchmark, and 8-token decode timing remains
too short for an optimization win claim.

GitHub Actions keeps correctness and performance separate. `verify.yml` runs
the functional suite on macOS ARM64, Ubuntu x64, and Windows x64; it does not
run the measured matrix or enforce speed thresholds. Existing recorded JSON
tests report formatting and mismatch rejection without running performance
samples inside TUnit.
`performance.yml` is a separate manually dispatched three-OS CPU matrix. It
fetches only the pinned Qwen smoke GGUF, builds the pinned competitors, runs
three warm-ups plus five measured fresh-process rounds with a two-thread cap,
and uploads unmodified per-round JSON for each runner. The C# `report --input
<raw.json> [--summary <path>] [--require-quality]` command renders workload, model digest,
platform, quality, median process wall time, peak RSS, Mac footprint, and
reported decode phase into the GitHub job summary. The quality flag fails a
job when output diverges, while the raw artifact is retained. Each runner is a distinct
cohort; this eight-token diagnostic neither replaces long generation nor
passes the 30-pair release benchmark gate. The first actual Actions run on
`9567d06` passed all three performance jobs and published three raw artifacts;
the quality-gate CLI change still requires a fresh workflow run.

Two 10-turn modes use `benchmarks/scenarios/travel-planner-10-turns.json`:

- **locked transcript** feeds the same pre-recorded assistant response into
  every next turn. It is the fair growing-context performance comparison.
- **live transcript** feeds each subject's generated response into its next
  turn. It is a realistic conversation test, but divergent context makes its
  latency numbers descriptive rather than paired.

The embedding scenario fixes documents, queries, batches, and repetitions.
Both runners are still planned. Cache cold/warm state, prefix reuse, model
residency, and session history must be explicit for every row.

The direct llama.cpp process is now a real diagnostic baseline even though
LLamaSharp already provides a llama.cpp-backed subject. Its native prompt-eval
and eval phases are not remapped to load/TTFT/generation columns. The current
five-sample native-only evidence is not paired with the earlier three-subject
Mac run; a formal direct-native verdict still requires interleaved paired
evidence. Optional
mistral.rs/DwarfStar subjects are eligible only when their pinned model and
no-Python execution path are compatible.

The native smoke tests are early evidence toward `AC-BMK-004-1` /
`TEST-BMK-004-1` (same workload) and `AC-PKG-006-3` / `TEST-PKG-006-3`
(native binary consumes the verified GGUF). They do not satisfy the complete
statistical/evidence contracts of `TASK-BMK-001` or `TASK-BMK-004`.

MLX and ONNX Runtime GenAI are additional candidate subjects, not substitutes
for the direct GGUF baseline. MLX may use an external native/Swift binary on
Apple Silicon; `mlx-lm`'s Python CLI is ineligible under the repository rules.
ONNX Runtime GenAI can be exercised from the C# benchmark project, but needs a
verified ONNX model package. A Metal MLX run and a CPU GGUF run are different
hardware cohorts; a separately exported or quantized ONNX/MLX model is a
different weight/precision cohort unless quality and provenance qualify it.
Unrun cells remain `not_run`; ONNX Runtime GenAI is not yet installed or
measured. The MLX cohort below has local diagnostic evidence but no confirmed
GitHub Actions measurement yet.

## Longer diagnostics and MLX Metal cohort (2026-09-28)

`dialogue` runs one- or three-turn locked-transcript scenarios from
`benchmarks/scenarios/`. The single request allows up to 128 output tokens;
the France/US/UK dialogue allows 64 per turn. It tokenizes each exact CPU
prompt once with the pinned GGUF tokenizer, then launches all four CPU
subjects in rotating order and retains actual generated counts, prompt IDs,
load/TTFT/decode (where available), process wall/CPU/RSS, derived average CPU
cores (process CPU time divided by wall time), and raw results.
`report-dialogue --input <raw.json> [--summary <path>]` renders these as a
separate performance report; neither command runs inside TUnit.
`performance.yml` now requests one warm-up and three measured rounds for each
longer scenario on every CPU runner. An early EOS is reported at its actual
output count. These diagnostic results are not the 30-pair release gate.

This CPU mode starts a **new process for every turn**. The locked earlier
assistant messages make prompt length grow deterministically but do not reuse
live KV. OS file-cache warming may occur and is not a KV-cache hit. The
separate `mlx` command instead starts a resident SwiftLM/MLX Metal server and
streams each request to measure client-observed TTFT, exact usage tokens,
request wall, CPU and whole-process RSS/physical footprint. Its pinned server
log can identify prompt-cache hit-token counts; absent hit logs are `n/a`, not
proof of a miss. Server load is separate from request wall. SwiftLM is an
external benchmark subject, not a Synapse engine, and uses a distinct pinned
MLX 8-bit SafeTensors Qwen package instead of the CPU GGUF Q8_0 weights.
Metal allocation is not independently measured; RSS and physical footprint
must not be added. Both cohorts currently label answer quality unreviewed.
The GitHub Actions MLX job downloads a SHA-256-verified prebuilt binary and
content-verified model; it does not build Swift or use Python.
