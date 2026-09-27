# Synapse

Synapse is a local-first C#/.NET inference engine with an optional Rust
acceleration layer. It owns model graph execution, memory and KV state,
scheduling, quantization, sampling, and
the worker protocol rather than hiding a third-party inference server.

The first managed inference slice is working on Apple Silicon: Synapse reads
the pinned Qwen2.5 Q8_0 GGUF through a read-only memory map and executes its
24-layer Qwen2 graph in C#, including RMSNorm, Q8_0 matrix/vector kernels,
GQA, RoPE, KV state, SwiGLU, and greedy sampling. The first-token result for
`The capital of France is` matches both required baselines (` Paris`, token
`12095`). This is correctness evidence, not yet a statistically qualified
performance claim.

## Constraints

- No Python or Node.js dependency anywhere in build, runtime, model tooling,
  tests, or benchmarks.
- ZoneTree is the embedded durable store for cache metadata and related state.
- dotLLM and LLamaSharp are required out-of-process comparison baselines.
- All performance claims require shared model inputs and raw paired evidence.

## Build

Prerequisites are .NET SDK 10.0.401 and Rust 1.98.1. On Apple Homebrew
installations, put `/opt/homebrew/opt/rustup/bin` first on `PATH`.

```text
dotnet restore Synapse.slnx --use-lock-file
dotnet build Synapse.slnx --configuration Release --locked-mode
dotnet test Synapse.slnx --configuration Release --no-build
cargo test --manifest-path native/Cargo.toml --locked
```

See `docs/Development/Commands.md` for the full verified command set and
`synapse.plan.md` for factual delivery status.

## Run the managed Qwen2 slice

The current CLI accepts token IDs while the repo-owned tokenizer is still on
the critical path. The five IDs below are the pinned model's encoding of
`The capital of France is`.

```text
dotnet run --project src/Synapse.Cli --configuration Release -- generate \
  --model tests/Fixtures/Models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf \
  --tokens 785,6722,315,9625,374 \
  --max-tokens 1 \
  --context-size 512
```

## Projects, dependencies, and attribution

Synapse keeps an explicit ledger of external work used to guide or exercise
the project.

| Project | How Synapse uses it | License / integration |
|---|---|---|
| [dotLLM](https://github.com/kkokosa/dotLLM) | Required pure-.NET CPU benchmark and architecture reference; pinned at `d88040451d7db56e5dfef9d5754ad0955b0f7fe5` | GPL-3.0; built and launched as a separate process. |
| [LLamaSharp](https://github.com/SciSharp/LLamaSharp) | Required llama.cpp-backed CPU/Metal benchmark through package `0.27.0` | MIT; package dependency of the benchmark process. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Native baseline reached through LLamaSharp and format/kernel literature reference | MIT; transitively used by the LLamaSharp benchmark. |
| [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | Shared Q8_0 fixture for dotLLM, LLamaSharp, and Synapse smoke/benchmark runs; SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | Apache-2.0 model artifact stored with Git LFS. |
| [ML.NET](https://github.com/dotnet/machinelearning) | GenAI pipeline, tokenizer, tensor, and `IChatClient` API reference | MIT; no runtime dependency yet. |
| [ZoneTree](https://github.com/ZoneTree/ZoneTree) | Required embedded durable store for cache metadata, prefix indexes, journals, and evidence indexes | Package dependency `1.9.8`. |
| [Aspire](https://github.com/dotnet/aspire) | Local orchestration, health, telemetry, and later real multi-process tests | Package/tool dependency when the AppHost slice begins. |
| [Microsoft Orleans](https://github.com/dotnet/orleans) | Later D3 cluster control plane for leases, placement, and recovery | Package dependency when the cluster slice begins. |
| [TUnit](https://github.com/thomhurst/TUnit) | .NET behavior test framework on Microsoft.Testing.Platform | Package dependency `1.70.1`. |
| [Rust](https://github.com/rust-lang/rust) | Optional native acceleration for measured C# hotspots | Toolchain dependency. |
