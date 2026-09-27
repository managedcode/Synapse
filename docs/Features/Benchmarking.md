# Benchmarking

The benchmark system launches Synapse, dotLLM, and LLamaSharp as isolated
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

Two 10-turn modes use `benchmarks/scenarios/travel-planner-10-turns.json`:

- **locked transcript** feeds the same pre-recorded assistant response into
  every next turn. It is the fair growing-context performance comparison.
- **live transcript** feeds each subject's generated response into its next
  turn. It is a realistic conversation test, but divergent context makes its
  latency numbers descriptive rather than paired.

The embedding scenario fixes documents, queries, batches, and repetitions.
Both runners are still planned. Cache cold/warm state, prefix reuse, model
residency, and session history must be explicit for every row.

The direct llama.cpp process remains a required baseline even though
LLamaSharp currently provides a llama.cpp-backed subject. Optional
mistral.rs/DwarfStar subjects are eligible only when their pinned model and
no-Python execution path are compatible.
