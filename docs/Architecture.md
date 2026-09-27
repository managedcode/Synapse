# Synapse architecture

Synapse is a local-first inference engine whose product code is owned in this
monorepo. C# owns the public SDK, Graph IR, model execution, memory and KV
state, orchestration, policy, diagnostics, durable metadata adapters, and the
initial worker data plane. Rust is an optional accelerator boundary used only
for hotspots whose profile proves the managed implementation is insufficient.

```text
.NET SDK / CLI / Host
        |
        | in-process locally; bounded typed protocol across workers
        v
C# graph runtime / worker ---- direct framed data ---- C# graph worker
        |
        +-- verified Graph IR and reference kernels
        +-- Q8_0 CPU execution and bounded hot KV
        +-- optional profiled Rust / Metal acceleration

ZoneTree (embedded, C#)
        +-- package/cache metadata
        +-- prefix index and state journal
        +-- benchmark/evidence catalog

Orleans (later D3 control plane only)
        +-- leases, placement, epochs, and recovery coordination
```

Large tensors and KV payloads never transit Orleans. Local execution never
requires Orleans, a coordination server, or a network. Unsupported models,
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
adds a bounded GGUF v3 reader and a managed Qwen2 Q8_0 forward/decode path. It
currently accepts pre-tokenized IDs; Graph IR extraction, the tokenizer,
Metal, paged KV, and the Orleans topology remain on the critical path.
