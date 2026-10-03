# ADR-023: rejected block-mean weight experiment

Status: Rejected and removed at the owner's request. Date: 2026-10-03.
Historical requirement/task: `REQ-QNT-006` / `TASK-QNT-006`; removed from
the active task registry. This decision supersedes its experimental acceptance.

A group mean discards the weight residual and changes the linear operator.
Sixteen synthetic probes on a real 576×576 SmolLM2 BF16 query matrix produced
relative output error energy 0.781296, compared with 0.015860 for Q4.
Synthetic timing and reconstruction did not qualify model quality.

A subsequent real Qwen2.5-0.5B Q8_0 retrieval pilot replaced adjacent groups
of 2/4/8 codes with rounded means. Every mean profile lost the answer while
the original FP32 and FP16 KV profiles answered correctly. Their tail NLL and
greedy traces showed severe drift. These variants retained the weight storage
and dense arithmetic; there was no useful memory or speed result to preserve.

Remove the grouped-mean codec/direct linear, weight-study command, mean
preparation options/manifest/gates, tests dedicated to those removed features
and generated mean models. Retain historical raw evidence as a rejected
experiment. Lossless preparation, ordinary quantization, sensitivity measurement
and precision selectors remain separate capabilities.

ZoneTree stays the required durable metadata/index store. Its lossless
compression and ordered storage do not make averaged neural weights equivalent.
Current work focuses on actual KV allocation, precision and exact prefix reuse
with full-model quality measurements (ADR-017 and ADR-025).
