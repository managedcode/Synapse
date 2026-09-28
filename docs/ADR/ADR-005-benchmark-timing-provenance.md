# ADR-005: external benchmark timing provenance

Status: Accepted. Date: 2026-09-28.

## Decision

`TASK-BMK-001` keeps each external subject out of the Synapse runtime. The
direct `llama-completion` process uses the same pinned GGUF, prompt, token
limit, context, greedy sampling, thread cap, and CPU-only mode as the existing
Qwen smoke. Its verbose prompt token IDs and generated text are quality gates.

The benchmark JSON distinguishes process-observed wall/CPU/working-set data
from subject-reported internal phases. `llama.cpp` prompt-eval and eval times
are recorded as native-only diagnostics; they are not renamed to end-to-end
TTFT or generation time. The native CLI does not expose a comparable model
load or first-token timestamp in this interface, so those fields are null.
Its eval rate is identified as native eval throughput, not placed in the
cross-subject end-to-end decode column. Missing metrics are never zero.

`llama-bench` is an additional, separate kernel-throughput control. It omits
tokenization and sampling and may not be merged with completion-process TTFT,
generation, or process-wall measurements. A result is comparable only when
revision, model digest, prompt token IDs, backend, hardware, and workload are
recorded together. Short smoke runs carry no winner verdict.

## Consequences

The benchmark result schema adds nullable timing fields and a provenance label
for native-only phases. Existing measured evidence remains unchanged. The
model catalog coverage table explicitly separates measured, not-run, and
unsupported rows. Neither an unavailable subject nor a missing model is
interpreted as a passing benchmark.

MLX and ONNX Runtime GenAI may enter the matrix as external subjects only via
Python-free launch paths, pinned model packages, and separate hardware and
weight/precision cohorts where appropriate. A cross-format label alone does
not establish comparable weights or tokenization.
