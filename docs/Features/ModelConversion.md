# Model conversion

`REQ-CNV-001`, `TASK-CNV-001`, ADR-024. Source preparation is an explicit
offline C# operation. Runtime opens the resulting `.synapse` file independently
of the original sources. No external model engine executes product graphs.

```text
synapse model formats
synapse model convert --source model.gguf --output model.synapse
synapse model convert --source graph.onnx --output graph.synapse --dimension batch=1:64
synapse model convert --source weights.safetensors --graph graph.json --output graph.synapse
synapse model inspect --model graph.synapse
synapse model run --model graph.synapse --inputs inputs.json
synapse generate --model model.synapse --tokens 1,2,3 --max-tokens 4
```

`model compile` remains the compatible GGUF-only command. `convert` dispatches
by explicit source format, never by brand name. Existing destinations are
preserved. Native graph packages use version 2; existing version-1 Qwen packages
remain executable by `generate` and `score`. Native ONNX/SafeTensors graphs
execute through `model run`; they do not imply an imported language-model
tokenizer, chat template, decoder state or generation adapter.

| Source | Implemented boundary | Explicit exclusions |
|---|---|---|
| GGUF | Lossless Qwen2 F32/Q8_0/Q4_K/Q6_K preparation using ADR-022 | Other families and backend restrictions remain explicit |
| ONNX | IR 7–10, default opset 13–21, FP32 constants/input/output; MatMul and restricted Gemm to Linear; equal-shape Add/Mul, Identity, last-axis Softmax | External/sparse tensors, custom domains/functions, control flow, broadcasts, arbitrary ranks and other operators |
| SafeTensors | Exact finite F32/F16/BF16 to FP32, explicit native graph JSON | Weights alone, arbitrary Hugging Face architecture/config/tokenizer import, unsupported dtypes |

ONNX input/value/output declarations must agree with inferred shapes. MatMul B
is a constant matrix; Gemm requires alpha=beta=1, transA=0, transB=0 or 1 and
an optional constant vector bias. Conversion materializes the explicit
`[out,in]` layout and reuses normalized weights. Unknown semantics fail before
publication. ONNX parsing uses bounded protobuf fields, not ONNX Runtime.

Native graphs support `Linear`, `Add`, `Multiply`, `Silu`, `Softmax`, `Identity`.
Every graph is checked for unique producers, acyclic dependencies, names,
supported operation arity and compatible shapes. Internal Identity aliases and
unreachable pure nodes/weights are removed. Public Identity output names remain
stable. These are exact structural passes, with no precision reduction,
averaging, layer skipping or approximate profile applied.

This sidecar defines one batched linear function over tensor `w`:

```json
{
  "schema_version": 1,
  "inputs": [{"name": "x", "shape": [
    {"symbol": "batch", "minimum": 1, "maximum": 64},
    {"symbol": null, "minimum": 2, "maximum": 2}
  ]}],
  "outputs": ["y"],
  "nodes": [{"name": "projection", "operation": "Linear", "inputs": ["x", "w"], "output": "y"}]
}
```

`w` must be a constant `[out,2]` matrix. Runtime input JSON is explicit:

```json
{"x": {"shape": [2,2], "data": [1,2,3,4]}}
```

Dynamic support is bounded shape binding: only the first axis of a rank-two
input can be symbolic; shared symbols use identical bounds and one binding.
All batch values use the same row count. Each independent row lowers to verified
Model IR and executes the permanent scalar interpreter with FP32
storage/compute and FP64 reductions. Rank-one vectors are also supported.
Input data is copied and must be finite. Unsupported shapes fail before math.
Cancellation is checked through import, publication, package verification and
the row loop, and before preflight, between interpreter nodes and before return.
Each bounded scalar operator finishes before its next cancellation boundary.
General dynamic scheduling, stateful/conditional graphs, GPU
graph lowering, MoE routing, fusion and automatic precision selection remain
planned in GraphExecution/FlyBrain; no throughput gain is claimed here.

Native package limits: 4 MiB manifest, 1,024 tensors, 1M elements per tensor and
16M aggregate declarations/output elements. ONNX source is limited to 64 MiB,
8M raw constant elements and 16M normalized elements. SafeTensors source is
limited to 128 MiB, its sidecar to 4 MiB, tensors to 16M aggregate elements,
and input/output declarations to 512 each. Native manifests, graph sidecars and
CLI input JSON reject duplicate/unknown fields and null required arrays.
Optional `symbol: null` denotes a fixed dimension. Malformed protobuf,
oversized counts and nonfinite data fail explicitly. SafeTensors header
descriptor compatibility follows the existing bounded container reader.

The 96-byte envelope binds the bounded native manifest and aligned FP32 payload
with SHA-256. The manifest retains source and sidecar hashes, graph fingerprint,
tensor ranges and executed pass names. Canonical ranges, zero padding, finite
weights and graph validity are checked on open. Source hashes are rechecked
before atomic publication. This proves integrity rather than publisher trust.

| Acceptance | Real regression coverage |
|---|---|
| `AC-CNV-001-1` / `TEST-CNV-001-1` source adapters | `OnnxImportTests`, `OnnxMalformedTests`, `SafeTensorConversionTests` |
| `AC-CNV-001-2` / `TEST-CNV-001-2` deterministic, independent package execution | `ModelConverterTests`, `NativeGraphPackageTests` |
| `AC-CNV-001-3` / `TEST-CNV-001-3` shape binding and exact graph passes | `ConversionGraphPipelineTests` |
| `AC-CNV-001-4` / `TEST-CNV-001-4` rejection, cancellation, publication safety | Package/adapter rejection cases and `ConversionGraphCancellationTests` real long-row/timer cancellation |
| `AC-CNV-001-5` / `TEST-CNV-001-5` explicit CLI preparation/run | `ConvertCliAcceptsExplicitSafeTensorsGraph`, CLI package regressions |

Exact commands and limitations are recorded in
`docs/Development/ModelConversionEvidence-2026-10-03.md`. This bounded slice
does not close general package import or FlyBrain runtime/benchmark tasks.
