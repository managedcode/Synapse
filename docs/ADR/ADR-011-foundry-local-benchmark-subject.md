# ADR-011: Foundry Local as an isolated benchmark subject

Status: Accepted. Date: 2026-09-28.

## Context

Microsoft Foundry Local runs catalog models on the local machine through ONNX
Runtime and ONNX Runtime GenAI. It is a relevant comparison for a .NET
inference engine, and it has its own model catalog with Qwen, Phi, Mistral,
and DeepSeek-R1-distilled checkpoints. ADR-005 already lets ONNX Runtime GenAI
enter the benchmark as an external subject when its launch path contains no
Python and its model packages are pinned.

The installed Homebrew CLI (`foundry` 0.8.119) is an older service-based
preview with a dynamic port. The current C# SDK, `Microsoft.AI.Foundry.Local`
2.0.1 (MIT), runs the same runtime in-process on macOS ARM64, Linux x64/ARM64,
and Windows x64/ARM64. A probe on the development Mac showed:

- `OpenAIChatClient.CompleteChatStreamingAsync` buffers the first chunks and
  returns no usage, so it cannot measure time to first token.
- `ChatSession` with `SetStreaming(true)` yields one text item per generated
  token and a final response with exact prompt and completion token counts.
- A `ChatSession` keeps its turns, so reusing one session would silently reuse
  conversation state.
- `RequestOptions.AdditionalOptions["repetition_penalty"]` has no observable
  effect. The package's `genai_config.json` search defaults therefore stay in
  force apart from temperature, sampling, and output length.
- The SDK's native runtime package sets `RuntimeIdentifier` from
  `buildTransitive`. A project that consumes it becomes RID-specific.
- Every catalog package sets `search.max_length` to the model's full context
  window with `past_present_share_buffer: true`, so ONNX Runtime GenAI
  allocates an FP32 KV buffer for that whole window at the first request. On
  the development Mac, Qwen3 0.6B (40,960 tokens) reached 4.9 GiB RSS and a
  9.6 GiB physical footprint with a 1.2 s first token; Phi-3.5-mini
  (131,072 tokens) reached 6.9 GiB RSS, a 98 GiB compressed footprint, and a
  12.4 s first token. Computed full-window KV sizes are 0.75 GiB (Qwen2.5
  0.5B), 8.75 GiB (Qwen3 0.6B), 8 GiB (Mistral 7B), 14 GiB
  (DeepSeek-R1 7B), 32 GiB (Phi-4-mini), and 96 GiB (Phi-3.5-mini). The Phi
  and DeepSeek packages cannot fit a 16 GB hosted runner that way, and
  Mistral would need about 13 GB of it.
  `AdditionalOptions["max_length"]` does not change the allocation. With
  `search.max_length` lowered to 1,024 in `genai_config.json`, the recorded
  Qwen3 run on the same 78-token prompt has a 100 ms first token and 1.5 GiB
  peak RSS. The package-default probes are kept as raw JSON
  (`*-package-default-context-probe.json`).

## Decision

1. **Separate project.** `experiments/Synapse.FoundryLocalBenchmarks` is the
   only project that references the SDK. The package reference is
   `PrivateAssets="all"`, so ONNX Runtime, ONNX Runtime GenAI, and the SDK's
   RID setting never flow into Synapse, the LLamaSharp runner, or the tests.
   The project pins `RuntimeIdentifier` to the host SDK RID and declares
   `osx-arm64;linux-x64;win-x64` so its lock file is identical on the three
   CI hosts. Foundry Local is a benchmark subject, never a runtime engine.
2. **Streaming session API.** Each request creates a new `ChatSession` with
   the full locked transcript (system, earlier user turns, locked assistant
   turns, current user turn). The model stays loaded in one process for the
   whole scenario. This is the resident-process, recomputed-transcript mode.
   No KV or prefix-cache reuse is claimed.
3. **Sampling is recorded, not assumed.** Requests set `Temperature = 0`,
   `DoSample = false`, and `MaxOutputTokens`. The evidence stores the
   package's `genai_config.json` search section verbatim, so a retained
   repetition penalty is visible.
4. **Context is bounded explicitly.** The set declares `contextTokens`
   (1,024). `fetch` saves the package's `genai_config.json` as
   `genai_config.original.json` and lowers only `search.max_length` to that
   bound. Weights and all other settings are unchanged. `run` exits with code
   4 when the bound is missing or different. The evidence records the
   effective bound, the original `max_length`, and the hashes of both config
   files. This mirrors the 512-token context of the GGUF cohorts. A run with
   the package default measures the preallocation, not generation, so it is
   not a scheduled benchmark.
5. **Threads are not capped.** SDK 2.0.1 does not expose ONNX Runtime thread
   settings. The evidence records `runtime_default_not_configurable`, the host
   processor count, and measured average CPU cores. This cohort is not
   thread-matched to the two-thread GGUF CPU cohort.
6. **No implicit download.** `fetch` is the only command that downloads a
   model. `run` exits with code 4 when the variant is not cached. The cache,
   app data, and logs live under one explicit directory, and non-essential
   telemetry is disabled.
7. **Pinned model set with a memory rule.**
   `benchmarks/model-sets/foundry-local-families.json` lists exact catalog
   variant IDs (the `:N` suffix is the catalog version), their catalog file
   sizes, the context bound, and the GitHub-hosted runners. A model is
   scheduled on a runner only when its file size is at most half of the
   runner's memory; with the bounded context the KV buffer is small. `run` fails when
   the live catalog reports a different size for the pinned variant. The
   evidence stores the SHA-256 of every model file.
8. **Isolation.** `plan` expands the set into one GitHub Actions matrix entry
   per runner and model. Each entry is a separate job on a fresh hosted
   machine. It downloads only its own model and measures it in one fresh
   process. Raw JSON is uploaded per job. Results from different runners, and
   from different models, are never pooled.
9. **Separate cohort.** Foundry models are ONNX packages quantized by
   Microsoft. Even the Qwen2.5 0.5B anchor has different weights, precision,
   and chat-template handling than the GGUF Q8_0 and MLX 8-bit cohorts. Its
   numbers are not ranked against them without quality and weight parity.

## Consequences

- One more NuGet dependency tree (SDK, ONNX Runtime 1.28.0, ONNX Runtime GenAI
  Foundry 0.15.2, Betalgo OpenAI models) is added, limited to one experiment.
- Catalog listing and model download need network access to the Foundry
  catalog. Measurement after load does not.
- The MiniMax family is not in the set. The Foundry catalog has no MiniMax
  model, and current MiniMax checkpoints do not fit a hosted runner.
- DeepSeek coverage is `deepseek-r1-distill-qwen-7b`, a Qwen2-architecture
  distillation. It does not exercise native DeepSeek MLA or MoE.
- Answer quality is `unreviewed`. A reasoning model can spend its whole
  output budget on reasoning text, which the evidence keeps separately.
- WebGPU variants are listed for local use only. Hosted runners have no GPU.
