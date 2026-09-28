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
Unrun cells remain `not_run`, and neither subject is currently installed or
measured by the CI benchmark matrix.
