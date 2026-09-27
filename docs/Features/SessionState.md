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
