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
