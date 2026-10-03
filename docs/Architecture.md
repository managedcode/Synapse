# Synapse architecture

Synapse is a local-first inference engine whose product code is owned in this
monorepo. C# owns the public SDK, typed IRs, portable/reference execution,
orchestration, policy, diagnostics, durable metadata adapters, and the first
implementation of every path. Rust takes optimized kernels, allocators, hot-KV
operations, and direct transfer only when a paired profile proves that boundary
is necessary. The portable C# path remains executable and tested.

```text
.NET SDK / CLI / Host
        |
        +-- in-process local graph runtime (no Orleans/network)
        |
        +-- Orleans request/control plane (planned D3)
                   | leases, epochs, region/weight-group placement
                   v
          graph worker ===== direct bounded data ===== graph worker
                   |
                   +-- verified Model IR -> Execution IR
                   +-- C# portable/reference kernels and KV semantics
                   +-- profiled Rust/native kernels, allocator, hot KV, transfer

ZoneTree (embedded, C#)
        +-- package/cache metadata
        +-- prefix index and state journal
        +-- benchmark/evidence catalog

Orleans (later D3 request/control plane only)
        +-- accepts/co-ordinates distributed requests
        +-- leases, placement, weight groups, epochs, and recovery
```

The central FlyBrain execution model is a bounded activation wave over coarse
regions. Model IR answers **what may execute**; Execution IR decides **how and
with which precision**; the deployment plan decides **where**. Region labels
are annotations only. A skippable region declares a graph decision or profile,
its structural/programmed/trained/approximate provenance, and explicit absent
or bypassed output behavior. See
`docs/ADR/ADR-002-flybrain-activation-waves.md`.

Large tensors, activations, and KV payloads never transit Orleans. There is no
grain per token, layer, node, or neuron. Local execution never requires
Orleans, a coordination server, or a network. Unsupported models,
operators, profiles, or devices fail before generation with structured errors.

Model preparation is an explicit operation before execution. The CLI converts
a source into a `.synapse` artifact; public generation, scoring and model
loading consume that prepared artifact. The initial Qwen2 package preserves
the exact source encodings, tokenizer metadata and graph identity in a bounded,
integrity-checked layout and executes directly from its mapped tensor ranges
(ADR-022). Source loaders remain internal numerical oracles. Preparation never
happens implicitly inside a measured runtime request. Grouped weight averaging
is a separate, unqualified approximation experiment (ADR-023); it is excluded
from default precision selection and generation.

## External comparison boundary

dotLLM, LLamaSharp, and direct llama.cpp are launched as separate benchmark
subjects. They are not product backends and cannot satisfy Synapse correctness
tests. Every comparison records its exact commit/package/native-binary
fingerprint and uses the same compatible model inputs. MLX and ONNX Runtime
GenAI are candidate external subjects only in qualified hardware/weight-format
cohorts. dotLLM source is GPL-3.0 and is not copied into this MIT repository.

## Initial vertical slice

The bootstrap doctor proves the installed .NET/Rust toolchains, current CPU
architecture and SIMD capability, memory budget validation, a real Rust child
process, and a durable ZoneTree write/reopen/read. The first inference slice
adds a bounded GGUF v3 reader, a verified 26-region Qwen2 Model IR, and a
managed Qwen2 Q8_0 forward/decode path. Text enters through a repo-owned
byte-level BPE tokenizer read from the GGUF vocabulary (ADR-014), qualified by
whole-corpus parity with `llama-tokenize`. The first scalar reference
operators and a fixed-shape dense tiny-graph interpreter are covered by FP64
and fail-closed tests. Qwen execution from the IR, paged KV, and the Orleans
topology remain on the critical path. GPU execution (ADR-012) runs a
brand-neutral dense-decoder layout through `native/synapse-gpu`: Metal is
implemented and tested on Apple silicon; CUDA shares the same step encoder and
ABI but has not run on NVIDIA hardware. Context limits and explicit YaRN
scaling are ADR-013. Quality is measured by teacher-forced scoring on every
backend (perplexity with the llama-perplexity protocol) and by exact-answer
long-context tasks, with every engine given identical tokens (ADR-015). Model
acquisition and import boundaries are in `ADR-004` and
`docs/Features/ModelPackages.md`.

Explicit `model convert` preparation (ADR-024) also lowers supported ONNX and
SafeTensors graphs to native version-2 `.synapse` packages. Bounded batch rows
execute locally through verified Model IR and the scalar reference interpreter.
The separate `model run` boundary exposes native tensor graph inputs/outputs;
text generation still requires a qualified decoder-family adapter. General
dynamic scheduling and optimized graph lowering remain planned.
