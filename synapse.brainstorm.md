# Synapse bootstrap decisions

## 2026-10-03 pipeline model parameters

Expose model IDs/files, model sets and smoke qualification as workflow inputs
and repository variables. Keep defaults visible in the workflow configuration,
and carry the selected values through fetch, cache, preparation and execution.
Ignoring generated `benchmarks/` output requires relocating the small tracked
scenario/configuration and recorded parser-fixture inputs used by clean CI.


## 2026-10-03 model preparation and performance investigation

- Compile supported Qwen GGUF sources into a deterministic, lossless,
  memory-mapped `.synapse` package with verified provenance and graph-ordered
  aligned tensors. This is an initial executable package slice; general
  SafeTensors family import and Execution IR compilation stay planned.
- Test four adjacent weights replaced by their mean as an explicitly
  approximate encoding. The direct linear computes one input-group sum and
  one multiplication per weight group. It needs measured distortion and
  held-out model quality before runtime use; arbitrary dense matrices do not
  have interchangeable weights.
- ZoneTree remains the durable metadata/index store. Its compaction and
  compressed storage inform immutable chunk/index layout; they do not prove
  that arithmetic averaging preserves model outputs.
- Remove per-element temporary arrays in scalar source decoding, preserving
  full validation before output writes. This targets model preparation, not
  token decode throughput.
- Profile the real local Metal K-quant path before changing kernels. Preserve
  paired raw measurements and qualify numerical parity; unavailable device
  access cannot be counted as passing hardware verification.

Decisions after measurement: keep exact whole-tensor source decoding with
SIMD finite validation (30 paired BF16 probes, 6.38735 -> 0.108675 ms,
identical bits and zero allocations); require explicit
preparation before runtime and reuse the loaded tokenizer. Leave four-weight
averaging outside executable profiles because the real-weight synthetic
output error energy is 0.7813 versus Q4's 0.01586. Revert all three slower
Metal GEMM pilots. Benchmark identity cannot be cached by path or file
timestamps: source and package are revalidated before each measured child.
The prepared package startup scan remains explicit; no generation-speed
claim follows from its layout alone.

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

## 2026-09-28 CPU kernels, concurrency, and cluster direction

- The profile placed the gap in the scalar Q8_0 linear and in per-row
  `Parallel.For`, not in model loading.
  - Fix: Q8_0 activations with `sdot`/AVX2 kernels, a persistent pool, fused
    regions, and batched prefill.
  - The Rust kernel won the weight-pass microbenchmark at one and two threads.
    Both kernels are memory-bound at eight.
  - The FP32 `reference` path stays as the oracle.
- A Q8-activation argmax flip at a 0.0115-logit near-tie is a numerical
  cohort difference, not a bug. Long exact-continuation gates need
  locked-continuation speed runs or an FP32-activation parity profile.
- Concurrency comes from continuous batching inside one model instance
  (ADR-007). Because decode is weight-bandwidth-bound, N requests share one
  weight pass.
- Cluster direction (ADR-009, owner): an Orleans cluster that nodes join,
  with one immovable grain per request.
  - Weights exist per node, per subset, or split per cluster.
  - Balancing uses silo-metadata filters, a residency-aware custom placement
    director, `ResourceOptimizedPlacement` as the fallback, and the
    rebalancer only for control grains.
  - New requests move to other nodes. Running requests and their KV do not.
- Rejected:
  - NativeAOT for hosts, because Orleans hosting is not a NativeAOT target.
  - Migrating running request grains, because hot KV would be stranded.
  - Splitting a model that fits one node across nodes, because it adds a
    network hop per stage per token without adding capacity.

## 2026-09-28 Foundry Local subject

- Measure Microsoft Foundry Local through its C# SDK 2.0.1 in-process, not the
  installed 0.8 CLI service. The SDK is current, MIT-licensed, and pinned by
  NuGet lock files. It supports all three hosted-runner platforms.
- Use `ChatSession` streaming. The OpenAI-style client buffers early chunks
  and omits usage. Create a new session per request, because a session keeps
  its turns.
- Keep the SDK in its own RID-specific project. Its native package forces a
  `RuntimeIdentifier`, which must not leak into the LLamaSharp runner or the
  test host.
- One model set covers Qwen, Phi, Mistral, and DeepSeek-R1-distill. The rule
  "file size at most half of runner memory" keeps every scheduled job inside
  a hosted runner. MiniMax is dropped: no catalog entry and no checkpoint that
  fits.
- Isolation is structural: one CI job per runner and model, one process per
  model, explicit download, raw JSON per job.
- Rejected: reusing one session per scenario (hidden KV reuse), the OpenAI
  client (no TTFT), and adding the SDK to the existing runner (RID leak).

## 2026-09-28 public benchmark interpretation

- Scope every public CPU claim to the preserved eight-token diagnostic: the
  native backend reached 102% of llama.cpp at two threads and 114% at eight,
  with exact eight-token parity. This is not a long-answer or 30-pair verdict.
- Treat the existing 274-token first-token gap as evidence from the older
  pre-batched-prefill build. The implementation changed; only a fresh run can
  confirm whether the bottleneck moved.
- Review Foundry output separately from throughput. The manual review found
  reasoning-only, factually wrong, and truncated answers; none of the seven
  model/device variants completed the 128-token instruction. Keep the raw
  evidence status unchanged and publish the review as a descriptive layer.

## 2026-09-28 hosted artifact reporting

- A successful matrix currently leaves 22 separate raw artifacts and no single
  run-level results table. Build one C# report from downloaded raw JSON after
  every measurement job, including failed or incomplete runs.
- Preserve distinct GGUF CPU, MLX Metal, and Foundry Local timing scopes. Show
  measured medians, sample counts, quality state, and missing artifacts. Upload
  the Markdown report as one artifact and show it in the final Actions summary.

## 2026-09-28 GPU backends and long context

- GPU host language. Options were C# Objective-C interop through
  `objc_msgSend`, a Swift or Objective-C shim, or Rust with `objc2-metal`.
  Swift and Objective-C are outside the allowed languages. C# interop would
  hand-roll every selector. Rust `objc2-metal` is typed and small, so the Metal
  host is Rust (ADR-012).
- Kernel delivery. Options were a prebuilt `metallib` from the Metal toolchain
  or source compiled at load. Runtime compilation removes a build-time Xcode
  component and keeps CI simple. CUDA follows the same rule with NVRTC, so no
  `nvcc` and no local NVIDIA download are needed.
- Model agnosticism. The first draft named the host after Qwen2. The owner
  rejected that: the GPU layer must not know brands or source formats. The ABI
  now describes a pre-norm dense decoder, and family adapters map tensor names.
  The ADR-004 package compiler will add a second adapter, and its chunk format
  can repack weights for GPU loads.
- Batch invariance. Matrix-vector decode is invariant: explicit `fma` order and
  templated token counts. GEMM and attention splits are not. Making attention
  splits position-determined with a sequential merge would restore
  bitwise-equal concurrency on GPU; this is kept as a plan item, not claimed.
- Long context. Trained limits now fail explicitly, and YaRN is an explicit
  profile. Pass-key retrieval, not synthetic-token speed, decides quality.
  - The 0.5B model kept 12 of 12 up to 65k.
  - At 120k it lost the 10%-depth key. The first llama.cpp control lost one
    token of the prompt (see the quality section below). On truly identical
    IDs, llama.cpp gave the same one-digit answer.
  - The largest measured engine gap is long-context prefill: llama.cpp was
    2x faster at 120k.

## 2026-09-28 long-context quality evaluation

The owner asked whether the model is correct on large prompts, not only fast,
and asked for comparisons across systems.

- Tokenizer first (ADR-014). Evaluation needs text in and text out. Options
  were an external tokenizer process, a Hugging Face `tokenizer.json`, or the
  GGUF arrays. The GGUF arrays are already mapped and pinned with the weights,
  so the tokenizer reads them. Parity is qualified over the whole repository,
  in both special-token modes, instead of a handful of strings.
- Engine fidelity. Options were KL divergence against llama.cpp
  `--kl-divergence-base` files, greedy-continuation agreement, or perplexity
  with the llama-perplexity protocol. A KL base file stores the full vocabulary
  per scored position: about 311 MB per 1,000 positions at 151,936 tokens, or
  roughly 20 GB for one 128k chunk. That is impractical at long context, so KL
  was rejected for now. Perplexity on identical tokens is cheap and uses the
  exact same protocol on both engines. Synapse also writes per-position traces,
  so its own profiles (FP32 against FP16 KV) are compared position by position.
- Model capability. An LLM judge was rejected: it adds a second model and
  non-determinism. RULER-style tasks were chosen instead, because their
  answers are exact and they are generated from a seed: a single needle, a
  multi-key needle with distractors, and variable tracking. A capability claim
  needs agreement between engines, because an answer every engine misses is a
  model limit.
- Two llama.cpp input transformations were found while doing this:
  - escape processing is on by default in every common-args tool;
  - `-f` drops one trailing newline from the prompt file.

  The second one invalidated the "identical token IDs" claim of the earlier
  120k pass-key control. That control evaluated 119,990 tokens, not
  119,991. The control is rerun with a compensated file, and the evidence is
  corrected instead of kept.
- MLX (SwiftLM) joins as a separate-weights cohort at 32k and below. Its
  `--ctx-size` option selects a rotating KV cache, which would silently drop
  early context, so it runs without that option. The chat API can only confirm
  the prompt token count, not the IDs.

## 2026-09-29 research: Metal kernels, long context, C#, and Orleans

Three research passes (web plus source reading; nothing copied) and our own
profiling. The owner asked whether C# and Orleans are worth keeping, how to
make large and dynamic context practical, and how to optimize the Mac first.

- **Where the time goes** (`SYNAPSE_GPU_PROFILE=1`, per-kernel GPU time):
  - At 4k the prompt GEMM takes 74% and attention 21%.
  - At 32k attention takes 70%.
  - Host code, C# and Rust together, costs 0.3% of wall time (38,380 ms of
    GPU time in a 38,490 ms prefill).
- **GEMM (open).** It runs at about 3.1 TFLOPS against an estimated 4 for
  llama.cpp. llama.cpp uses the same 64×32 tile and simdgroup geometry.
  - Five experiments each moved the result by 5% or less: half weights,
    half weights plus half activations, contiguous 8×8 blocks, grid order,
    and explicit unrolling.
  - The epilogue costs 5%.
  - 64×64 tiles spilled (3–4× slower).
  - llama.cpp's concurrent dispatch is worth only 3–5% to llama.cpp itself.
  - The remaining gap needs shader-profiler counters (occupancy, spills), not
    more guesses.
- **Attention (improved).**
  - `simdgroup_matrix::thread_elements()` is typed as the whole 8×8 vector,
    so indexing it made the compiler spill. The lane's pair must be
    reinterpreted as `float2`, as MLX does.
  - Keeping scores and outputs as plain `float2` registers and building
    matrices only for each multiply removed the threadgroup round trip and
    the diagonal rescale.
  - 32k time to first token went from 32.6 s to 27.0 s (llama.cpp 26.6 s,
    MLX 15.5 s).
  - Two eight-row halves per simdgroup share each K/V load but spilled
    (38.1 s), so one half is used.
- **Long context.**
  - Qwen2.5-0.5B is rated for 32k; YaRN ×4 to 128k is documented only for 7B
    and up. The practical long-context target is Qwen2.5-7B-Instruct-1M
    (native to 256k, 56 KiB/token), which needs a download and head
    dimension 128 on the GPU.
  - Dynamic YaRN or NTK needs pre-RoPE keys or re-rotation, so a fixed
    factor per session is preferred.
  - The biggest wins, in order:
    1. an MLX-level prefill kernel;
    2. a persistent prefix cache (ZoneTree metadata, mmap payloads), which
       turns a repeated 128k document from minutes of prefill into I/O;
    3. block-sparse prefill (XAttention/FlexPrefill style) above 32k;
    4. Quest-style decode with absolute 2–4k token budgets, keeping the
       first layers dense;
    5. fused q8_0 KV.
  - Sliding windows and eviction lose information and conflict with prefix
    reuse (SCBench).
- **Dynamic context (ADR-017).**
  - Done: KV grows on demand.
  - Configuring 131,072 tokens with a 4k prompt now peaks at 234 MiB (FP16)
    against llama.cpp's 1,645 MiB with the same `-c`.
  - Paged KV blocks are the next form: no copies, and the page as the unit
    of Quest bounds, quantization, and disk persistence.
- **C#.** No speed cost on the GPU path (0.3%), and on CPU the Rust kernels
  do the hot work.
  - Value: the .NET SDK and NuGet surface, the portable tested reference
    path, and the tooling.
  - Cost: two languages, an FFI boundary, and a kernel set in C#, Rust, and
    Metal.
  - Verdict: keep C# if a .NET-consumable engine is a product goal. A
    Rust-only host would be simpler otherwise.
- **Orleans** (AGENTS.md makes it the control plane; changing that is the
  owner's call).
  - It adds nothing on one machine, and the design already keeps it off the
    local path.
  - On several machines, pipeline parallelism buys capacity, not
    single-stream speed. llama.cpp RPC drops from 20.4 to 15.2 tok/s on 1→4
    nodes (Qwen3-235B, Jeff Geerling). Tensor parallelism helps only over
    RDMA (Thunderbolt 5), and this M2 Pro has Thunderbolt 4.
  - No comparable system (exo, llama.cpp RPC, distributed-llama, MLX
    distributed, Petals, prima.cpp) uses an actor framework. vLLM is
    reducing Ray to a launcher.
  - Orleans' storage-backed membership and default failure detection
    (about 90 s) fit 2–8 pinned multi-GB stages poorly.
  - Recommendation: keep Orleans out of the data path (as today) and defer
    it until a measured multi-machine, multi-user workload needs it. First
    experiment: a two-node pipeline of a 7B model over Thunderbolt 4 and
    10 GbE, measuring per-hop latency and tok/s against one node.

## 2026-09-29 a model that is not loaded whole

Owner direction: the model should not load whole; parts can be dropped. Orleans
stays (owner decision). The owner commits.

- **Three ways to run less of a model, measured on Qwen2.5-7B-Instruct-1M:**
  - *Layer drop* (ADR-019): a shallower model whose dropped weights and KV
    are never loaded. Honest result: an unhealed dense checkpoint has few
    cheap layers. Two of 28 cost nothing in perplexity but flip 20% of
    top-1 predictions; four cost +39%. Memory falls by about 240 MiB per
    layer.
  - *Speculative decoding* (ADR-020): the 0.5B drafts and the 7B verifies.
    The output is exactly the 7B's, with +23–35% decode on chat text at 3
    draft tokens. It pays only where the draft agrees often.
  - *Quantization* (Q4_K_M in llama.cpp): 4,742 MiB against 8,096 and 29.2
    against 18.6 tokens/s at a small quality cost. Per quality point it
    saves the most memory, so Q4_K/Q6_K kernels are the next step.
- **Rejected for now:**
  - Self-speculation with a layer-dropped 7B draft. Dropping 8 of 28 layers
    keeps 71% of the cost and agrees on only half the tokens.
  - Streaming a dense model's layers from SSD under a memory budget (F2
    capacity mode). Every token touches every layer, so this reads about 8
    GB per token.
- **Later, where "not loaded" is exact:** MoE expert paging (F5, OLMoE),
  where unselected experts are outside the function.
- **Update the same day (ADR-021):** Q4_K/Q6_K now run on Metal. The 7B Q4_K_M peaks at 4.2 GB in
  Synapse (llama.cpp 4.8 GB) with no measured perplexity loss against Q8_0. Decode is 0.64× llama.cpp's, which is
  the next kernel target.
# Format-neutral conversion checkpoint (2026-10-03)

The user wants one explicit source-to-`.synapse` slice covering GGUF, ONNX and
other formats, with native graphs and dynamic capabilities. Start with GGUF,
ONNX and SafeTensors; weight-only sources require an inert graph description.
Separate source adapters, exact graph preparation, artifact publication and
execution. Preserve version-1 Qwen packages while adding a native graph v2.
Bound dynamic batches and execute real reference math before describing support.
Identity/dead-node elimination is safe; adaptive scheduling, GPU lowering,
MoE/stateful graphs and approximate profiles require their own qualification.
