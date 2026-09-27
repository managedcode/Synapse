# Benchmarking

The benchmark system will launch Synapse, dotLLM, and LLamaSharp as isolated
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

These are source fingerprints, not measured results. No winner is declared
until raw paired evidence exists. The initial common model candidate is the
small SmolLM GGUF used by dotLLM's documented examples; its exact repository,
revision, file hash, tokenizer/template, and license must be frozen before the
first run.
