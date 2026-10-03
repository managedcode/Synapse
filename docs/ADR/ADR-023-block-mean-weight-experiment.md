# ADR-023: explicit block-mean weight experiment

Status: Accepted for an isolated experiment. Date: 2026-10-03.
Requirement: `REQ-QNT-006`. Task: `TASK-QNT-006`.

## Context

The owner proposes replacing four adjacent weights with one average to reduce
weight traffic and arithmetic. For a group, the arithmetic mean minimizes
unweighted squared reconstruction error, but discarding the residual changes
the linear operator. In general, `sum(w[i] * x[i])` differs from
`mean(w) * sum(x[i])`. This is a lossy structural approximation and is distinct
from lossless package compression.

For independent zero-mean weights, averaging a group of four retains only one
of its four degrees of freedom. Calibration and held-out model quality must
decide whether a tensor can tolerate this approximation. A weight-only error
or a faster isolated matrix multiply cannot qualify a model.

ZoneTree uses lossless block compression, ordered indexes, caches, and LSM
maintenance to accelerate persistent data access. Those mechanisms are useful
for package/chunk metadata and cache policy; they do not preserve neural
network behavior after unequal weights are averaged.

## Decision

- Add an experimental C# `BlockMeanCodec`, separate from default runtime
  encoding discovery. Its identity is
  `syn.approx.blockmean.g{group}.f32.v1`; the default group has four weights,
  and supported group sizes are 2 through 1024.
- A row stores one little-endian FP32 arithmetic mean per adjacent group.
  The final partial group averages only its real elements. There is no stored
  padding. Size is `rows * ceil(columns / group) * 4` bytes.
- The encoder validates all weights and shapes before writing, rejects
  overlapping source/destination storage, sums in FP64, and rounds each mean
  once to FP32. Finite source values cannot overflow their arithmetic mean.
- The decoder validates every stored mean before writing and rejects
  overlapping encoded/output storage. Each mean expands across its logical
  group. NaN and infinity are invalid; either sign of finite zero is valid.
- Direct linear first sums each input group in FP64 once, then dots every row
  of means with those sums. SIMD may reorder FP64 accumulation; comparisons
  use a documented tolerance against dense FP64 arithmetic on the expanded
  weights. It allocates no expanded matrix.
- The ordinary linear entry rents scratch buffers. A caller-owned scratch
  overload enables repeated allocation-free measurement. Scratch must not
  alias inputs, encoded bytes, output, or other scratch. Output may alias
  input or encoded storage because it is published only after all results
  are computed. Invalid means, non-finite inputs, and FP32 output overflow
  leave output unchanged.
- The experiment does not select this encoding in executable model profiles.
  Promotion requires separate calibration and held-out data, full-model
  perplexity and task-quality gates, explicit approximation permission,
  backend support, and raw paired end-to-end speed/memory evidence.

## Verification

`TEST-QNT-006-1..8` cover the explicit experiment identity, golden mean bytes,
partial groups and exact sizes, constant-group exactness, dense/direct linear
parity, unchanged output on rejected data, alias behavior, and caller scratch
validation. A C# real-weight diagnostic reports reconstruction/output
distortion and paired timings without interpreting probe vectors as real
model activations or claiming model quality.

No runtime, real-model quality, or benchmark release task is completed by this
decision.
