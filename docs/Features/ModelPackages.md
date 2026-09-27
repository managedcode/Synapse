# Model packages

## Outcome

Synapse separates reproducible acquisition from executable support. The
catalog can safely download an architecture before an importer or kernel
exists, but every status surface must say which layer is available:

1. **cataloged** — immutable source files, sizes, hashes, and license exist;
2. **download verified** — the local bytes match the catalog;
3. **container indexed** — bounded GGUF/SafeTensors metadata was parsed;
4. **model imported** — config, tokenizer/template, tensors, and graph match a
   supported family;
5. **executable** — required operators and state semantics are implemented;
6. **benchmark eligible** — output parity and provenance gates pass.

Only Qwen2 Q8_0 currently reaches executable/benchmark-eligible smoke status.
SmolLM2 and the BERT embedding fixtures currently reach bounded SafeTensors
index verification. Other catalog entries are acquisition targets, not hidden
claims of runtime support.

## Catalog sets

- `smoke`: the smallest current end-to-end generation gate.
- `family-small`: small dense decoder families used while expanding import and
  tokenizer parity.
- `architecture-small`: inexpensive but structurally distinct architectures,
  currently Qwen3 and attention-free Mamba.
- `embedding-small`: two BERT-family retrieval encoders.
- `medium`: models that fit the 32 GB development Mac and expose Phi3,
  distilled Qwen2, and Mistral3 behavior under realistic memory pressure.

Native DeepSeek MLA/MoE and MiniMax hybrid/MoE are tracked separately from
their brand names. DeepSeek-R1-Distill-Qwen-1.5B is a Qwen2 checkpoint. The
current native MiniMax families are too large to be honest local smoke
fixtures. A small model is added only when it brings a distinct architecture,
not to inflate the model count.

## Trust and failure behavior

The rules are defined by `ADR-004`. Important failures happen before model
allocation: unsafe package/file path, unsupported schema, duplicate IDs/files,
non-HTTPS URL, untrusted redirect, redirect overflow, unexpected length,
digest mismatch, oversized SafeTensors header, shape overflow, overlapping
ranges, and unknown dtype.

Temporary files use a unique suffix and become visible at the destination only
after verification. An interrupted or rejected download cannot be mistaken for
a valid cached artifact.

## Next implementation slice

`TASK-PKG-002..005` adds the actual Synapse manifest/chunk format, tensor source
descriptors, repo-owned BPE/chat templates, SafeTensors conversion, ZoneTree
content identity cache, and atomic compiler/export. It must not execute remote
model code or introduce Python/Node tooling.
