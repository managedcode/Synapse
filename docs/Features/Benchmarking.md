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
`performance.yml` is a separate three-OS CPU matrix that runs on every push
to `main` and supports manual dispatch. It fetches only the pinned Qwen smoke
GGUF, builds the pinned competitors, runs
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

## Foundry Local subject (ADR-011)

Microsoft Foundry Local is measured by
`experiments/Synapse.FoundryLocalBenchmarks`, a separate RID-specific C#
project that references `Microsoft.AI.Foundry.Local` 2.0.1. It keeps the
model loaded in one process and sends every locked-transcript request through
a new streaming `ChatSession`. It records exact prompt and output token
counts, TTFT, decode rate, request wall time, process CPU time, whole-process
peak RSS, and macOS physical footprint. It also records the SDK and ONNX
Runtime versions, the variant's execution provider, the package's
`genai_config.json` search defaults, and the SHA-256 of every model file.

The model set is `benchmarks/model-sets/foundry-local-families.json`:

| Family | Alias | Catalog CPU variant | File MB | macOS 15 (7 GB) | Ubuntu / Windows (16 GB) |
|---|---|---|---:|---|---|
| Qwen | `qwen2.5-0.5b` | `qwen2.5-0.5b-instruct-generic-cpu:4` | 822 | scheduled | scheduled |
| Qwen | `qwen3-0.6b` | `qwen3-0.6b-generic-cpu:4` | 593 | scheduled | scheduled |
| Phi | `phi-3.5-mini` | `Phi-3.5-mini-instruct-generic-cpu:2` | 2,590 | scheduled | scheduled |
| Phi | `phi-4-mini` | `Phi-4-mini-instruct-generic-cpu:5` | 4,915 | excluded, memory | scheduled |
| Mistral | `mistral-7b-v0.2` | `mistralai-Mistral-7B-Instruct-v0-2-generic-cpu:3` | 4,167 | excluded, memory | scheduled |
| DeepSeek | `deepseek-r1-7b` | `deepseek-r1-distill-qwen-7b-generic-cpu:4` | 6,584 | excluded, memory | scheduled |

A model is scheduled only when its catalog file size is at most half of the
runner's memory. "Scheduled" is a plan, not a measurement; a cell becomes
evidence only after its job uploads raw JSON. MiniMax is absent: the Foundry
catalog has none and current MiniMax checkpoints exceed every hosted runner.
The DeepSeek entry is an R1 distillation into Qwen2 7B.

Commands (`dotnet experiments/Synapse.FoundryLocalBenchmarks/bin/Release/net10.0/Synapse.FoundryLocalBenchmarks.dll`):

- `plan --set <set.json> [--summary <path>]` prints a one-line GitHub matrix
  with one entry per runner and model, and a coverage table with exclusions.
- `fetch --set <set.json> --alias <alias> --cache <dir>` is the only command
  that downloads. It also applies the set's context bound (below).
- `run --set <set.json> --alias <alias> --cache <dir> --scenario <json>
  --output <new.json> [--max-tokens 64] [--warmups 1] [--measurements 3]`
  exits with code 4 if the variant is not cached.
- `report --input <raw.json> [--summary <path>]` renders medians per turn.

`performance.yml` runs a `foundry-local-plan` job, then one isolated
`foundry-local` job per matrix entry. `verify.yml` fetches only the Qwen2.5
0.5B anchor and runs the functional TUnit checks against it.

Context bound: every Foundry package sets `search.max_length` to its full
context window, and ONNX Runtime GenAI allocates an FP32 KV buffer for the
whole window at the first request. With package defaults, Qwen3 0.6B reached
4.9 GiB RSS and a 1.2 s first token on the Mac; Phi-3.5-mini (131,072 tokens)
reached a 98 GiB compressed footprint and a 12.4 s first token. The set
therefore declares `contextTokens: 1024`. `fetch` keeps the original config as
`genai_config.original.json` and lowers only `max_length`; `run` refuses a
model without that bound. This matches the 512-token context of the GGUF
cohorts. Weights are unchanged.

Limits: the SDK does not expose ONNX Runtime thread settings, so this cohort
is not thread-capped. Its weights and quantization differ from GGUF Q8_0 and
MLX 8-bit, so it is never ranked against those cohorts. Raw evidence retains
the original `quality_unreviewed` status and stores reasoning apart from
answer text. A later manual documentation review found that none of the seven
local model/device variants fully completed the 128-token instruction: two
were reasoning-only, two contained material factual errors, and the remaining
three omitted or truncated required sections. That review is descriptive and
does not make the run eligible for a quality-adjusted verdict.

### Requirements

- `REQ-BMK-002`: Foundry Local is an isolated external benchmark subject with
  a pinned SDK, a pinned multi-family model set that fits hosted runner
  memory, explicit downloads, and raw per-job evidence.

| Criterion | Test |
|---|---|
| `AC-BMK-002-1` the plan schedules each model only on runners where it fits the memory rule, emits one matrix entry per runner and model, and covers Qwen, Phi, Mistral, and DeepSeek | `TEST-BMK-002-1` `FoundryPlanSchedulesOneIsolatedJobPerRunnerAndModel` |
| `AC-BMK-002-2` an invalid model set (duplicate alias, a model that fits no runner) is rejected | `TEST-BMK-002-2` `FoundryPlanRejectsInvalidModelSets` |
| `AC-BMK-002-3` `run` refuses an uncached variant and downloads nothing | `TEST-BMK-002-3` `FoundryRunRefusesUncachedModelWithoutDownloading` |
| `AC-BMK-002-4` a real run records provenance, the effective context bound and the package's original `max_length`, exact token counts, timing, CPU, and whole-process memory for every request; a model without the bound is refused | `TEST-BMK-002-4` `FoundryRunRecordsRealStreamingEvidence` |
| `AC-BMK-002-5` the report renders a separate cohort with no cross-cohort ranking | `TEST-BMK-002-5` `FoundryReportRendersSeparateCohortFromRecordedEvidence` |
| `AC-BMK-002-6` CI runs one Foundry job per plan entry and the functional suite fetches only the anchor model | `TEST-BMK-002-6` `WorkflowsIsolateFoundryLocalJobs` |
