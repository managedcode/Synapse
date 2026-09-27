# ADR-001: bootstrap and external boundaries

Status: Accepted. Date: 2026-09-27.

## Decision

Bootstrap with a real C#/Rust offline doctor and durable ZoneTree probe. Keep
dotLLM and LLamaSharp outside the product dependency graph as benchmark
processes. Defer Aspire until a real topology exists.

## Consequences

The repository proves both product languages and the required embedded store
before model execution work. A baseline can fail or be unavailable without
silently becoming a Synapse backend. GPL dotLLM code is never copied into the
MIT product. The first AppHost commit must include a runnable resource graph.

## Verification

`TEST-BST-003-1..3` cover CPU discovery, typed budget rejection, real Rust
process execution, ZoneTree reopen durability, and non-zero test suites.
