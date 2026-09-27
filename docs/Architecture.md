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
are annotations only. Skipping work requires an explicit graph predicate,
trained policy hash, or evaluated approximation profile. See
`docs/ADR/ADR-002-flybrain-activation-waves.md`.

Large tensors, activations, and KV payloads never transit Orleans. There is no
grain per token, layer, node, or neuron. Local execution never requires
Orleans, a coordination server, or a network. Unsupported models,
operators, profiles, or devices fail before generation with structured errors.

## External comparison boundary

dotLLM and LLamaSharp are launched as separate benchmark subjects. They are
not product backends and cannot satisfy Synapse correctness tests. Every
comparison records their exact commit/package/native-binary fingerprint and
uses the same compatible model inputs. dotLLM source is GPL-3.0 and is not
copied into this MIT repository.

## Initial vertical slice

The bootstrap doctor proves the installed .NET/Rust toolchains, current CPU
architecture and SIMD capability, memory budget validation, a real Rust child
process, and a durable ZoneTree write/reopen/read. The first inference slice
adds a bounded GGUF v3 reader, a verified 26-region Qwen2 Model IR, and a
managed Qwen2 Q8_0 forward/decode path. It currently accepts pre-tokenized IDs;
execution from the IR, the tokenizer, Metal, paged KV, and the Orleans topology
remain on the critical path. Model acquisition and import boundaries are in
`ADR-004` and `docs/Features/ModelPackages.md`.
