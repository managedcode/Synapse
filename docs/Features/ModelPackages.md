# Model packages

Explicit format-neutral conversion is implemented as a bounded separate slice
in [ModelConversion](ModelConversion.md), ADR-024. It adds ONNX/SafeTensors
native graph packages; the Qwen GGUF preparation below remains version 1.

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

Qwen2 Q8_0 reaches executable/benchmark-eligible smoke status; Q4_K/Q6_K
have qualified reference/Metal execution (ADR-021). Product execution requires
separate compilation into `.synapse` (ADR-022).
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

## Source tensor decoders

`SourceFormats` decodes every stored source encoding that Synapse accepts into
FP32 reference values. On-the-fly quantization and the reference executor
consume those values.

- Floating point: F32, F64 (must fit FP32), F16, BF16, FP8 E4M3FN, and FP8
  E5M2, used for both GGUF and SafeTensors dtypes.
- GGUF blocks of 32: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, and MXFP4.
- GGUF 256-value super-blocks: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS,
  TQ1_0, and TQ2_0.

Every block is validated before anything is written. A NaN or infinite scale,
an MXFP4 or Q8_K scale that would overflow FP32, or a NaN/infinite float value
raises `SourceEncodingException` and leaves the destination unchanged.

GGUF types without a verified decoder throw `UnsupportedType` and name the
ggml type:

- Q8_1, which ggml uses for activations.
- The grid-coded IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S. They
  need ggml's lattice tables, which will be imported from pinned MIT llama.cpp
  source with attribution rather than retyped.
- NVFP4 and Q1_0.
- Integer tensor types.

FP8 decoding covers the element values only. A model's block-scale recipe
(for example a separate `weight_scale_inv` tensor) belongs to its family
adapter.

Verification:

- Golden blocks computed by hand for Q4_0, Q5_0 (high bits), Q8_0, F16/BF16,
  and FP8 anchors.
- FP8 E5M2 is compared against the upper byte of an IEEE half for every finite
  code.
- Differential tests (`GgmlReferenceDecoderTests`) call ggml's own
  `dequantize_row_*`, `ggml_fp16_to_fp32_row`, and `ggml_bf16_to_fp32_row` from
  the native library shipped with the LLamaSharp benchmark subject. They use 16
  random blocks per type. ggml serves only as the format's test oracle, never at
  runtime.
  - Multiplicative formats match bit for bit.
  - Affine formats (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) match within 2⁻²⁰ of the
    largest value, because ggml builds may fuse `a*b − c`.
  - Block sizes and type names are also checked against `ggml_blck_size`,
    `ggml_type_size`, and `ggml_type_name`.
- A deliberate sign mutation in Q4_K was caught by the differential test
  before being reverted.

Source preparation now avoids per-element arrays for all scalar formats.
F32/F16/BF16 decode whole tensors after a complete finite-value scan using
SIMD exponent masks where available; the first invalid index is preserved.
BF16 uses SIMD widening on disjoint little-endian storage. F32 uses a bulk copy
when storage is disjoint. Portable fallbacks and existing forward alias
semantics remain. `AC-PKG-003-4` / `TEST-PKG-003-4`
(`ScalarDecodingAllocatesNoMemoryPerElement`, `LateNonFiniteScalarLeavesWholeDestinationUnchanged`)
guard allocation and rejection behavior. `AC-PKG-003-5` / `TEST-PKG-003-5`
(`WholeTensorFloatDecoderTests`) guard finite F32 bit patterns, every finite
BF16/F16 code, signed zero/subnormals, invalid index/output atomicity,
length/overflow and aliases. The 21 whole-tensor cases also pass with hardware
intrinsics disabled. Decoder timing is a preparation diagnostic,
separate from model loading, compilation and generation throughput.

## Next implementation slice

The bounded first `TASK-PKG-004` compiled artifact is described below.
`TASK-PKG-002..005` still includes general manifests/content-addressed chunks,
multi-family tokenizer/template import, SafeTensors conversion, and ZoneTree
content identity caching. It must not execute remote model code or introduce
Python/Node tooling.

## Explicit lossless preparation (`REQ-PKG-004`)

Decision: `ADR-022`. Runtime entry points require a prepared artifact. Convert
once, inspect the result, then use the compiled path:

```text
synapse model compile --source model.gguf --output model.synapse
synapse model inspect --model model.synapse
synapse generate --model model.synapse --tokens 1,2,3 --max-tokens 4
```

The first compiler supports the current executable Qwen2 family and preserves
F32/Q8_0/Q4_K/Q6_K tensor bytes, shape, encoding, and the complete GGUF metadata
header, including tokenizer arrays. It verifies the existing Model IR and
packs tensors in that graph's first-use order with 64-byte alignment. Original
source tensor ranges and basename remain graph provenance, so source and
compiled graph fingerprints agree. Runtime opens the compiled file itself
through a read-only memory map and uses existing family executors; Qwen
execution from Model IR/Execution IR remains planned.

The format has a 96-byte little-endian envelope: eight-byte `SYNAPSE\0` magic,
32-bit version, 32-bit manifest length, 64-bit payload offset, 64-bit total
file length, and SHA-256 digests of manifest and payload. The inert JSON
manifest is limited to 4 MiB; the verbatim source header to 64 MiB; source
metadata and tensor counts to 16,384. Tensor offsets are relative to the
payload; ranges are canonical, aligned, non-overlapping, and checked against
the preserved source index. The manifest records source SHA-256, graph
fingerprint, original ranges, and source encoding. Package identity hashes
the envelope. Digests provide integrity, not publisher authentication.

Compilation streams with one reused 64 KiB copy buffer, checks cancellation,
rechecks source identity after copying, flushes to disk, verifies the output,
then moves a unique temporary sibling atomically. It never overwrites an
existing destination. Failure deletes the partial artifact. Unsupported
families, unknown versions/types, malformed bounds, mismatched hashes, and
inconsistent source/compiled indexes fail explicitly. Compilation changes no
weight values and applies no averaging or approximate compression.

Complete payload verification adds an intentional startup scan. The format
alone is not a demonstrated throughput improvement; any speed claim requires
paired measurements and unchanged output quality.

| Criterion | Regression |
|---|---|
| `AC-PKG-004-1`: deterministic, lossless tensor and metadata packing with source digest | `TEST-PKG-004-1`: `CompilationPreservesTensorBytesArraysAndSourceIdentity`, `CompilationIsDeterministicAcrossDestinationNames` |
| `AC-PKG-004-2`: direct execution without source, equal graph identity/logits/tokens on reference/managed/native/available Metal | `TEST-PKG-004-2`: `PackageExecutesDirectlyWithoutSourceAndMatchesEveryBackend`, `PackagePreservesKQuantReferenceOutputs` |
| `AC-PKG-004-3`: reject malformed index/envelope/payload, including recomputed adversarial digests | `TEST-PKG-004-3`: `ReaderRejectsEnvelopeAndPayloadCorruption`, `ReaderRejectsMalformedIndexEvenWhenDigestsAreRecomputed` |
| `AC-PKG-004-4`: no publication on invalid or unsupported sources, cancelled writes, identical paths, or failed move | `TEST-PKG-004-4`: `CompilerRejectsInvalidSourceRangesWithoutPublishing`, `CompilerRejectsUnsupportedSourcesExplicitly`, `FailedPublicationAndCancellationPreserveExistingDestination`, `CompilerRejectsIdenticalPathsWithoutChangingSource`, `AtomicMoveFailureDeletesCompiledTemporaryFile` |
| `AC-PKG-004-5`: separate explicit CLI preparation; runtime rejects unprepared sources | `TEST-PKG-004-5`: `CompileCommandPublishesExecutablePackage`, `RuntimeRequiresExplicitCompiledArtifact` |
| `AC-PKG-004-7`: output decoding reuses the loaded mapping and a single thread-safe cached tokenizer | `TEST-PKG-004-7`: `LoadedTokenizerDecodesWithoutReopeningCompiledPath`, `UnsupportedLoadedTokenizerStillReturnsNullCliText` |
| `AC-PKG-004-6`: public runtime rejects source formats without conversion | `TEST-PKG-004-6`: `RuntimeRequiresExplicitCompiledArtifact` |
| `AC-PKG-004-10`: mixed-engine benchmark requires a matching prepared artifact and rechecks each sample | `TEST-PKG-004-10`: `BenchmarkRefusesUnpreparedSourceWithoutCreatingOutput`, `BenchmarkVerifiesPreparedArtifactAgainstSourceIdentity`, `BenchmarkRevalidatesChangedFilesAfterSuccessfulPreparation` |

General compressed chunks, ZoneTree cache integration, SafeTensors compilation,
per-tensor quantization and a graph scheduler
remain outside this implemented slice. Hardware that cannot run a parity case is
`not_run_missing_hardware`, never passed.

Focused evidence on 2026-10-03: the initial 31 compiled-package cases passed on macOS
ARM64, including actual reference, managed, native, and Metal execution, with
zero skips. The sandbox-only run exposed no Metal device and failed that case;
the rerun with actual device access passed all 31. Scoped whitespace format
and diff checks passed. The final 33 cases include loaded tokenizer reuse and
strong ordered byte/token/logit assertions. Full .NET verification passed
429/429; focused package file coverage is 232/251 (92.43%). Exact commands,
raw evidence and the ZoneTree background-exception limitation are recorded in
`benchmarks/results/2026-10-03-performance-preparation-evidence.md`.

The pinned Qwen2.5-0.5B Q8_0 source (675,710,816 bytes, SHA-256
`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`)
compiled into 675,749,760 bytes with 291 tensors. Inspection passed and real
managed CLI execution produced the same tokens `[12095,13]` and text ` Paris.`.
One sequential startup diagnostic measured 100.2118 ms in the saved GGUF
baseline and 593.863 ms for the compiled artifact. This exposes startup
verification cost, not a statistical performance comparison: cache state and
binary revisions differ. Raw local diagnostics remain under ignored
`artifacts/performance-audit/`, including `compiled-package-qwen05b.json`,
`gguf-load-diagnostic.json`, and `compiled-load-diagnostic.json`.
