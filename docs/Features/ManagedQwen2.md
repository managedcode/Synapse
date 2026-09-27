# Managed Qwen2 inference

## Working slice

The C# runtime reads GGUF v2/v3 metadata and tensor indexes with bounded counts,
validates the Qwen2 dimensions it consumes, and maps tensor payloads read-only.
The initial executor supports this fixture's F32 vectors and Q8_0 matrices.
It performs token embedding lookup, pre-norm attention with Q/K/V biases,
NeoX RoPE, grouped-query causal attention, an in-memory KV cache, residuals,
SwiGLU feed-forward layers, final RMS normalization, and greedy argmax.
The verified Model IR carries normalization epsilon, RoPE theta/layout/head
dimension, attention heads/scale/mask, and scalar decode position explicitly;
RoPE, KV append, and attention all consume the same position entry input.
KV slot shapes use the GGUF model limit as symbolic
`Context[1..model_max_context]`; the concrete per-instance KV and scratch
buffers still use the caller's bounded session context.

The pinned correctness probe uses prompt tokens
`785,6722,315,9625,374`. Synapse, dotLLM at commit
`d88040451d7db56e5dfef9d5754ad0955b0f7fe5`, and LLamaSharp `0.27.0`
all produce token `12095`, decoded as ` Paris` by the model tokenizer.

## Current boundaries

- Input is pre-tokenized until the repo-owned BPE tokenizer lands.
- The executor is the first vertical implementation and still needs to be
  lowered behind the specification's typed Graph IR and verifier.
- KV is bounded per model instance but is not paged, persisted, or recoverable.
- CPU Q8_0 is implemented; Q4 and Metal are not yet implemented.
- The observed timings are development evidence, not a benchmark verdict.
