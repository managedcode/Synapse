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

The memory headline is the operating-system observation of the **whole subject
process**, including CLR and native allocations: peak resident working set on
macOS, Linux, and Windows, and physical footprint on macOS when `proc_pid_rusage` permits
it. Sampling begins before model load and continues through generation; the
report records the sample count and separately labels private/virtual address
space, which is not resident memory. Synapse CLI and LLamaSharp additionally
report CLR live heap and cumulative managed allocation as diagnostics. These are subsets or
allocation-flow counters, not a partition of resident bytes. In particular,
`resident - GC heap` is **not** reported as native allocation: mmap pages,
shared libraries, runtime heaps, and driver memory make that subtraction
invalid. A missing platform metric remains null with its scope recorded.

Functional CI and performance evidence are separate workflows. Functional
TUnit checks can validate rendering against recorded raw evidence, but do not
run the measured matrix or assert throughput. The manually
dispatched performance workflow runs its own macOS/Linux/Windows matrix and
retains every measured round as an artifact with a descriptive report. An
output mismatch fails the job after writing the report, while the raw artifact
is still uploaded. Hosted
runners are variable hardware, so their results are never pooled across OSes
or compared to the owner's Mac as a release winner verdict.

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
