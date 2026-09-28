# Session state and ZoneTree ownership

C# owns live KV tensors, allocation leases, page references, append position,
and rollback fences. A future native accelerator may operate on leased buffers
but does not become their authority. ZoneTree is required for durable cache-adjacent state:
prefix fingerprints, page/object metadata, committed output cursors, snapshot
catalogs, and evidence indexes.

Persisted keys include the model source hash, effective weight/profile hash,
tokenizer/template hash, input-token hash, precision, layout version, and
tenant/isolation scope. A mismatch is a cache miss, never a best-effort reuse.
ZoneTree is not used as distributed consensus; the later authority layer owns
leases and epochs.

## Local KV slots (ADR-007, 2026-09-28)

The first executable session model is in-process. A loaded Qwen2 instance
owns a bounded pool of FP32 KV slots, `[layer][position][kvWidth]`:
- Slot 0 serves the synchronous `Generate` path.
- Slots `1..MaximumConcurrentSessions` serve `GenerateAsync` requests through
  the continuous-batching scheduler. That scheduler thread is the only writer.

A slot belongs to one request until the request completes or is canceled.
Positions restart at zero, and attention reads only positions `0..p` of its
own slot, so a reused slot cannot expose earlier tokens
(`CancelledSlotReusedSafely`). These slots are neither paged nor persisted,
and ZoneTree indexes nothing for them yet. Paged, branchable KV remains
`kv-performance.plan.md`. Cluster ownership (a request grain per request,
node-local replicas) is ADR-009.
