# ADR-022: lossless compiled Qwen2 model packages

## Status

Accepted on 2026-10-03 for the bounded first slice of `REQ-PKG-004` and
`TASK-PKG-004`. General family import, Execution IR scheduling, profiles,
content-addressed chunk deduplication, and cache integration remain planned.

## Decision

The C# compiler emits a single versioned `.synapse` file containing an inert,
size-bounded JSON manifest, a verbatim bounded GGUF header, and tensors packed
in first-use order from the verified Qwen2 Model IR. Payload starts and tensor
starts are aligned to 64 bytes. Source encodings and bytes are preserved;
compilation performs no averaging, quantization, layer removal, or numerical
approximation. The supported executable source types remain F32, Q8_0,
Q4_K, and Q6_K, subject to each existing backend's restrictions.

The manifest binds the complete source SHA-256, source length and basename,
original tensor ranges, graph fingerprint, and compiled tensor offsets. An
envelope SHA-256 binds the manifest; a payload SHA-256 binds the preserved
header, alignment bytes, and tensor payload. Identity is derived from those
digests. Digests detect corruption; they do not authenticate untrusted models.
Opening validates schema, bounds, alignment, encoding, source-index agreement,
tensor uniqueness, non-overlap, graph fingerprint, and payload integrity before
mapping the compiled file read-only. Metadata arrays are read directly from
the preserved header inside that mapping. No source extraction occurs.

Graph weight descriptors retain original source ranges and basename so
physical repacking does not change the source graph fingerprint. The runtime
uses separately rebased mapped tensor offsets. This verifies the existing
Qwen graph and calls the existing Qwen executors; it does not claim that Qwen
execution is scheduled from Model IR or Execution IR.

All writes and hashes stream through bounded buffers with cancellation.
Compilation creates a unique temporary sibling, flushes and verifies it, then
publishes atomically with a non-overwriting move. Source and output paths must
differ. Failure or cancellation deletes the temporary file and preserves any
existing destination. The compiler rejects unsupported families and excess
headers rather than introducing a hidden interpreter or whole-file buffer.

Runtime entry points require a separately prepared `.synapse` artifact. The
user runs `synapse model compile --source model.gguf --output model.synapse`
before loading or generating. Runtime startup never implicitly converts a
source model. The internal GGUF execution entry is explicitly reserved for
compiler validation and regression oracles; external baseline subjects keep
their own GGUF interchange requirements.

Tensor packing is a concrete preparation step, not a demonstrated throughput
improvement. Complete payload integrity hashing can increase startup time
relative to loading a GGUF source directly. Performance claims require paired
raw measurements and unchanged output quality.

`ITextGenerationModel.CreateTokenizer()` reads tokenizer arrays from the
already verified model mapping, with one cached tokenizer per loaded model.
It preserves the family's existing tokenizer support checks. CLI output
decoding uses this loaded-model capability instead of reopening the package
and hashing its complete payload a second time. Initial package integrity
verification remains mandatory; unsupported tokenizer fixtures still report
unavailable text explicitly through the existing CLI null output behavior.
The read-only compiled handle permits atomic rename/deletion sharing, while
verification and mapping use the same handle. Replacing a pathname therefore
does not replace the loaded model's already verified bytes.

## Verification

`TEST-PKG-004-*` uses deterministic real tiny Qwen fixtures, compares source
and compiled graph fingerprints, bytes, token outputs, and logits on actual
reference, managed, native, and available Metal backends. It covers deterministic
compilation, preserved metadata arrays, malformed envelopes/indexes, corrupt
payloads, truncation, unsupported sources, cancellation, and failed publication.
Missing GPU hardware remains `not_run_missing_hardware`.

`TEST-PKG-004-7` verifies loaded tokenizer vocabulary/text parity, reuse of the
same tokenizer, and output decoding after a real tiny tokenizer-capable
compiled fixture is renamed while its already validated mapping remains open.
