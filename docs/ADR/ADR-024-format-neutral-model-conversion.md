# ADR-024: format-neutral model conversion

Accepted 2026-10-03 for `REQ-CNV-001`, `TASK-CNV-001`.

The explicit C# `model convert` pipeline accepts GGUF, ONNX and SafeTensors.
GGUF delegates to the lossless Qwen2 compiler (ADR-022). ONNX and SafeTensors
use a version-2 native graph package under the same `SYNAPSE\0` envelope.
Version 1 remains readable. Generation and scoring still require preparation.

The version-2 manifest owns named FP32 inputs, tensor shapes, graph operations,
output names, aligned tensor ranges, original source hashes and the exact
conversion passes applied. It does not preserve an external executable or
delegate execution to ONNX Runtime. SafeTensors requires an explicit inert
graph sidecar because weights alone do not define a function. No remote code,
Python, Node, pickle or implicit model download is permitted.

The initial executable graph subset is Linear (weights `[out,in]`, optional
vector bias), same-shape Add/Multiply, SiLU and last-axis Softmax. Identity
elimination and unreachable-node elimination are exact structural passes.
ONNX MatMul/Gemm normalize constant weight layouts. Unknown operators, custom
domains, external tensors and unsupported numeric recipes fail explicitly.

Rank-one vectors and rank-two batches are supported. A symbolic batch dimension
requires explicit positive finite bounds; equal symbols share one binding.
Execution validates all input bindings and intermediate shapes before work,
then runs each independent row through verified Model IR and the permanent
scalar reference interpreter. This is bounded shape binding, not an adaptive
graph scheduler, GPU lowering, MoE routing or a throughput claim. Those remain
in the GraphExecution/FlyBrain plans. FP32 storage and FP64 reference reductions
are explicit; no averaging, layer drop or quantization is implicit.

The scalar interpreter accepts an optional cancellation token. Conversion
forwards its caller's token into each row; the interpreter checks it before
preflight, between graph nodes and before returning outputs. An individual
bounded scalar operator finishes atomically before the next cancellation
boundary. This preserves its numerical order while allowing cancellation
during a long single-row graph, rather than only after the entire row.

Parsing, graph/tensor counts, bytes and element counts are bounded. Package
envelope/manifest/payload hashes, canonical aligned ranges, graph validity and
finite weights are checked. Publication is an atomic non-overwriting move;
cancellation and failures remove temporary output. Source identities are
checked before and after conversion. Runtime needs only the prepared package.

`TEST-CNV-001-*` covers real protobuf/SafeTensors files, real graph execution,
dynamic binding, deterministic output, exact passes, malformed inputs and
publication failures. Unsupported hardware never counts as passed.
