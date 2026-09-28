# Synapse bootstrap decisions

Date: 2026-09-27. Status: accepted for the bootstrap slice.

## Decisions

1. Start with one real offline vertical slice rather than generating every
   future project. The slice includes a typed C# doctor, a native Rust doctor,
   a real process boundary, and a durable ZoneTree round trip.
2. Pin the exact toolchains observed on the development machine: .NET SDK
   10.0.401 and Rust 1.98.1. Package revisions are locked by NuGet and Cargo
   lockfiles after restore.
3. Use ZoneTree for durable cache metadata and cache-adjacent local state.
   Tensor payloads and hot KV pages remain owned by the C# runtime; ZoneTree is
   not a substitute for the bounded allocator and is not consensus.
4. Treat dotLLM and LLamaSharp as out-of-process benchmark baselines. This
   avoids contaminating the runtime dependency graph and respects dotLLM's GPL
   boundary. The first shared real-model case uses the pinned Qwen2.5 0.5B
   Q8_0 GGUF accepted by all three engines.
5. Defer Aspire AppHost creation until there is a real host/worker topology to
   model. An empty AppHost would violate the specification's vertical-slice
   and no-placeholder rules.

## Rejected alternatives

- Making Rust the primary engine was rejected. C# owns execution first; Rust
  remains available only for a hotspot that profiling proves too slow.
- A custom embedded store was rejected because ZoneTree is a direct product
  requirement and already supplies the required durable ordered-key/value
  foundation.
- Linking dotLLM into product code was rejected because it would make the
  baseline a hidden engine and create GPL distribution implications.

## 2026-09-28 benchmark audit: direct native control

- Add pinned `llama-completion` as a third external GGUF subject. Run it in a
  fresh process with CPU-only flags, explicit prompt/context/thread/sampling
  settings, and verify the actual prompt token IDs and continuation.
- Preserve timing boundaries. Its reported prompt-eval and eval times are
  internal native phases; startup/model load is measured only by process wall
  time. Missing comparable load/TTFT fields are null, never zero or a renamed
  native phase. `llama-bench` is a separate kernel-throughput diagnostic.
- Use one model-by-subject coverage table for the whole pinned catalog. A model
  without a successful compatible run shows `not_run` or `unsupported`, not an
  extrapolated throughput value. An eight-token smoke is too short for a
  performance verdict; add a longer-output diagnostic before formal 30-pair
  evidence.
- Add MLX and ONNX Runtime GenAI to the candidate set without a Python or Node
  dependency. MLX's external Swift/native subject belongs to a macOS Metal
  cohort; ONNX Runtime GenAI's C# subject requires a pinned ONNX package.
  Neither may inherit the GGUF Q8_0 numbers: different model formats or
  quantizations require weight provenance and a separate cohort unless
  numerical/quality parity is demonstrated.

## 2026-09-28 whole-process benchmark memory checkpoint

- Use process-wide OS resident/footprint peaks as the comparable CPU memory
  envelope for all subjects, regardless of managed or native implementation.
  Do not confuse private virtual address space with physical consumption.
- Expose CLR heap/allocated bytes only where the measured .NET subject can
  report them without attaching a diagnostic agent. Never infer native bytes
  by subtracting heap size from RSS or footprint.
- Record sampling coverage and missing metric status. A memory claim needs a
  fresh process from pre-load to exit, the same workload, and raw paired runs;
  the current short Qwen smoke remains diagnostic only.

## 2026-09-28 CI performance separation

- Keep functional `verify` fast and assertion-oriented. A long, exact Qwen
  continuation is not portable evidence of a quality regression when CPU
  kernels choose a different valid sequence. Validate report and quality-gate
  behavior from preserved raw evidence in TUnit; run the real matrix only in
  the performance workflow.
- Dispatch performance independently on macOS, Linux, and Windows. Preserve
  each runner's raw process samples and render a descriptive summary of the
  workload and medians. Hosted-runner results form separate hardware cohorts;
  they are not a cross-OS leaderboard or a substitute for paired owner-Mac runs.

## 2026-09-28 longer workload follow-up

- Preserve the eight-token run as startup/parity smoke only. Add a 128-token
  single request and three 64-token locked-transcript turns (France/Paris,
  United States/Washington, United Kingdom/London) with actual generated-token
  counts and prompt IDs in raw evidence. A token cap is not a promise that a
  model will keep generating until it; early EOS remains visible.
- Keep cold fresh-process growing-context numbers distinct from resident
  model, warm-prefix, and hot-KV experiments. Current CLIs lack a common
  persistent-session API, so no cache win can be asserted from this runner.
- For MLX, prefer a verified prebuilt Apple Silicon binary as requested. Its
  Metal/MLX-format model stays a separate cohort from GGUF Q8_0; capture
  release digest and model revision and never infer quality/speed parity from
  the shared Qwen family name.
