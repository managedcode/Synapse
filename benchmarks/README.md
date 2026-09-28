# Benchmark details

The short version is in the [main README](../README.md#benchmark). This page
lists every recorded run, how it was measured, and its raw data.

Runs A–K and N–O use Qwen2.5 0.5B Instruct, the only model Synapse runs today. Run L
measures Microsoft Foundry Local alone on six models from four families.

## Runs

| Run | Synapse version | Threads | Rounds | Raw data |
|---|---|---:|---|---|
| A. 128-token answer, 4 CPU engines | SIMD C# kernels (in progress) | 2 | 1 measured, no warm-up | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-single-128-diagnostic.json) |
| B. 3-turn dialogue, 4 CPU engines | SIMD C# kernels (in progress) | 2 | 1 measured, no warm-up | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-3turn-diagnostic.json) |
| C. MLX on the GPU, answer + dialogue | Not included (MLX only) | GPU | 1 warm-up + 3 measured | [answer](results/2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-single-128-diagnostic.json), [dialogue](results/2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-3turn-diagnostic.json) |
| D. 8-token smoke test, 3 engines | First scalar C# | 12 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-smoke.json) |
| E. 8-token memory run, 4 engines | First scalar C# | 8 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-four-subject-memory-clr-smoke.json) |
| F. GitHub Actions, 3 operating systems | First scalar C# | 2 | 3 warm-up + 5 measured | [workflow run](https://github.com/managedcode/Synapse/actions/runs/36405097376) |
| G. llama.cpp alone | Not included | 12 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-native-llamacpp-qwen2.5-0.5b-q8_0-diagnostic.json) |
| H. 32-token output check | First scalar C# | 8 | 1 measured, no warm-up | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-32tok-quality-divergence-final.json) |
| **I. 8-token, newest (main README)** | Rust kernels | 2 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-2thread-smoke.json) |
| **J. 8-token, newest** | Rust kernels | 8 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-native-8thread-smoke.json) |
| **K. 8-token, newest** | SIMD C# kernels | 2 | 3 warm-up + 5 measured | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-cpu-kernels-managed-2thread-smoke.json) |
| L. Foundry Local, 6 models, answer + dialogue | Not included (Foundry only) | runtime default (about 6 cores) | 1 warm-up + 3 measured | 14 JSON files + 2 default-context probes, `results/2026-09-28-m2-pro-foundry-local-*` |
| M. Hosted CPU, MLX, and Foundry, 3 operating systems | Rust CPU kernels for Synapse | CPU 2 / external runtime default | CPU smoke 3+5; long/MLX/Foundry 1+3 | [GitHub run `36435838291`](https://github.com/managedcode/Synapse/actions/runs/36435838291), 22 raw artifacts |
| N. Metal GPU long context, 40k and 131k synthetic tokens | Metal backend (ADR-012), FP32 KV | GPU | 1 measured per row | [JSON](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-long-context-speed-diagnostic.json) |
| O. Metal GPU pass-key retrieval, 4k to 120k | Metal backend, FP32 KV (native window) and FP16 KV (YaRN ×4) | GPU | 1 measured per row | [native](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-passkey-native-f32-diagnostic.json), [YaRN](results/2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-metal-passkey-yarn4-f16-diagnostic.json) |

Local machine: MacBook Pro, Apple M2 Pro (8 performance + 4 efficiency cores,
19-core GPU), 32 GB, macOS 27.0 arm64. Every CPU sample starts a new process.
Prompts are in [`scenarios/`](scenarios/). Notes on runs A–C:
[`results/2026-09-28-long-dialogue-and-mlx-evidence.md`](results/2026-09-28-long-dialogue-and-mlx-evidence.md).

Rules for reading the numbers:

- Runs are separate groups. Do not compare numbers across different runs.
- `n/a` means the engine does not report that value. It is not zero.
- Peak RSS is the whole process: managed, native, and memory-mapped model pages.
- None of these runs is a final ranking. The release gate needs 30 paired runs.

## I–K. Newest CPU kernels (8-token answer)

Prompt `The capital of France is`, median of 5 runs, 4 engines rotated each
round. All engines produced the same 8 tokens in every run. Power state was
not recorded.

| Engine | I. Rust, 2 threads | J. Rust, 8 threads | K. SIMD C#, 2 threads | Whole request (I) | Peak RSS (I) |
|---|---:|---:|---:|---:|---:|
| Synapse | 136.6 tok/s | 190.6 tok/s | 124.4 tok/s | 216 ms | 555 MiB |
| llama.cpp | 133.7 tok/s | 167.2 tok/s | 141.8 tok/s | 690 ms | 1,256 MiB |
| LLamaSharp | 97.1 tok/s | 147.0 tok/s | 100.1 tok/s | 945 ms | 1,275 MiB |
| dotLLM | 9.7 tok/s | 31.7 tok/s | 9.7 tok/s | 2,227 ms | 1,190 MiB |

First token in run I: Synapse 32 ms, LLamaSharp 32 ms, dotLLM 844 ms.
llama.cpp reports only its internal prompt time: 22 ms. llama.cpp speed is
its internal generation rate.

## L. Microsoft Foundry Local, 6 models (separate test)

Foundry Local SDK 2.0.1 (ONNX Runtime 1.28.0, ONNX Runtime GenAI 0.15.2) with
Microsoft's own ONNX packages from the Foundry catalog. These are not the
Q8_0 file used in runs A–K, and the SDK does not let us set a thread count:
the process used about 6 of the 12 cores. Do not compare these numbers with
runs A–K.

Each model and scenario ran in a fresh process. The model stayed loaded for
1 warm-up and 3 measured requests; every request was a new chat session with
the full transcript. Context was capped at 1,024 tokens (see below). Greedy
decoding. AC power; other desktop apps were open (load average about 3.4).
The raw schema still says `quality_unreviewed`; the manual review below was
performed afterward from the preserved text and reasoning fields and does not
retroactively change measurement eligibility.

128-token answer (`capitals-single-long`), median of 3:

| Model | Family | Device | File MB | Load ms | Prompt tokens | First token ms | Speed tok/s | Request ms | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B | Qwen | CPU | 822 | 832 | 78 | 65 | 231.4 | 614 | 650 |
| Qwen3 0.6B * | Qwen | CPU | 593 | 1,176 | 78 | 100 | 124.3 | 1,122 | 1,493 |
| Phi-3.5-mini | Phi | CPU | 2,590 | 2,878 | 81 | 448 | 42.1 | 3,467 | 3,563 |
| Phi-4-mini | Phi | CPU | 4,915 | 4,353 | 72 | 489 | 28.6 | 4,943 | 4,413 |
| Mistral 7B v0.2 † | Mistral | CPU | 4,167 | 5,570 | 84 | 821 | 25.2 | 5,857 | 5,132 |
| DeepSeek-R1 7B * | DeepSeek | CPU | 6,584 | 5,957 | 68 | 685 | 22.0 | 6,458 | 5,193 |
| Qwen2.5 0.5B | Qwen | GPU (WebGPU) | 700 | 773 | 78 | 45 | 142.3 | 736 | 809 |

Three-turn dialogue (`capitals-france-us-uk-3-turns`, 64 tokens per turn),
turn 3, median of 3:

| Model | Device | Prompt tokens | First token ms | Speed tok/s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|
| Qwen2.5 0.5B | CPU | 293 | 237 | 223.9 | 988 |
| Qwen3 0.6B * | CPU | 293 | 362 | 116.9 | 1,736 |
| Phi-3.5-mini | CPU | 295 | 1,607 | 39.9 | 3,672 |
| Phi-4-mini | CPU | 272 | 1,878 | 28.5 | 4,657 |
| Mistral 7B v0.2 † | CPU | 307 | 3,064 | 24.5 | 5,270 |
| DeepSeek-R1 7B * | CPU | 269 | 2,625 | 21.8 | 5,571 |
| Qwen2.5 0.5B | GPU (WebGPU) | 293 | 113 | 144.1 | 1,008 |

- \* Qwen3 and DeepSeek-R1 write reasoning text before the answer. Every
  generated token is counted; the raw JSON keeps reasoning text apart.
- † Mistral 7B v0.2's chat template has no system role, so the system prompt
  opens the first user message (`system_prompt_mode` in the JSON).
- The GPU run used 0.5–0.6 CPU cores instead of 6. Its 128-token answer
  stopped at 99 tokens.
- Prompt token counts differ because each model has its own tokenizer and
  chat template.

### Manual output review

No model/device variant fully satisfied the 128-token instruction. This makes
the throughput table useful for runtime diagnosis, not for a quality-adjusted
model ranking.

| Variant | Review of the deterministic measured output |
|---|---|
| Qwen2.5 0.5B CPU | Incorrect and incomplete: it names all three capitals, but calls Fort Knox the federal headquarters and misidentifies Paris landmarks; it hits the 128-token cap before the recap. |
| Qwen2.5 0.5B WebGPU | Mostly factual but incomplete: it stops at 99 tokens without the requested three-line recap. |
| Qwen3 0.6B CPU | Incorrect and incomplete: all 128 tokens are reasoning, it emits no final answer, and the reasoning calls Paris, London, and Madrid France's capitals. |
| Phi-3.5-mini CPU | Incorrect and incomplete: it corrupts Washington, D.C., places Mount Rushmore near it, and reaches the cap during the recap. |
| Phi-4-mini CPU | The three main country-capital sections are substantially correct, but the generated recap is malformed and truncated at the cap. |
| Mistral 7B v0.2 CPU | The France section is correct, but the output reaches the cap during the United States section before covering the United Kingdom or recap. |
| DeepSeek-R1 7B CPU | Incomplete: all 128 tokens are reasoning and no final answer is emitted, although the reasoning recalls the three capitals. |

The three-turn outputs have the same constraint: every CPU turn reaches its
64-token cap. DeepSeek-R1 and Qwen3 remain reasoning-only. Phi-3.5, Phi-4, and
Mistral contain mostly correct briefing content but truncate sections; the
Qwen2.5 variants additionally contain errors such as placing Paris in the
center or south of France and describing Washington transit as free. No
instruction-compliance claim is made from those dialogue timings.

**Context cap.** Every Foundry package sets its context length to the model's
full window, and ONNX Runtime GenAI reserves the KV memory for that whole
window at the first request. With the package default, Qwen3 0.6B reached
4.9 GiB RSS and a 1.2 s first token, and Phi-3.5-mini (131,072 tokens) reached
a 98 GiB compressed footprint and a 12.4 s first token. Raw data:
[Qwen3](results/2026-09-28-m2-pro-foundry-local-qwen3-0.6b-cpu-package-default-context-probe.json),
[Phi-3.5-mini](results/2026-09-28-m2-pro-foundry-local-phi-3.5-mini-cpu-package-default-context-probe.json)
(one 32-token request each). The runner's `fetch` therefore lowers only
`search.max_length` to 1,024 and keeps the original config beside it.

**GitHub Actions.** The performance workflow runs one isolated job per runner
and model: a model runs only where its file is at most half of the runner's
RAM. That is 3 models on `macos-15` (7 GB) and all 6 on `ubuntu-24.04` and
`windows-2025` (16 GB), 15 jobs in total. All 15 jobs passed in
[run `36435838291`](https://github.com/managedcode/Synapse/actions/runs/36435838291)
and retained separate raw answer/dialogue JSON. Their raw quality state remains
unreviewed; a green job confirms execution and artifact delivery, not answer
correctness.

## M. Hosted current-kernel run (`36435838291`)

The [performance run](https://github.com/managedcode/Synapse/actions/runs/36435838291)
passed all 20 jobs and retained 22 raw artifacts: three eight-token CPU
matrices, three CPU answer/dialogue pairs, one MLX answer/dialogue pair, and
15 Foundry model/runner answer/dialogue pairs. Each row below is a median of
measured rounds on that runner; warm-ups are excluded. CPU wall time includes
a fresh process and model load. MLX and Foundry request wall time excludes
their resident model load and belongs to separate weight/runtime cohorts.

| CPU runner | 8-token Synapse / llama.cpp wall ms | 8-token Synapse / llama.cpp RSS MiB | 128-token Synapse / llama.cpp wall ms | 128-token Synapse / llama.cpp reported decode tok/s |
|---|---:|---:|---:|---:|
| macOS M1 | 526 / 1,671 | 551 / 1,203 | 5,856 / 4,223 | 32.5 / 59.0 |
| Ubuntu x64 | 537 / 635 | 553 / 724 | 4,503 / 3,608 | 36.5 / 47.0 |
| Windows x64 | 557 / 681 | 539 / 575 | 4,936 / 3,330 | 30.3 / 52.0 |

All measured eight-token outputs matched. The 128-token and three-turn CPU
outputs retain `quality_status: unreviewed`; their timings are not eligible for
a quality-adjusted winner verdict. Hosted MLX produced a 128-token answer at
73.0 tok/s and 1,799 ms request wall on the M1, with three different measured
continuations. Hosted Foundry scheduled 3/6/6 models on Mac/Ubuntu/Windows;
Qwen3 and DeepSeek again spent the 128-token answer budget in reasoning without
a final answer. Foundry and MLX are not ranked against the GGUF CPU subjects.

The workflow's final job downloads every raw artifact and publishes one
per-run Markdown table with runner, scenario, turn, subject, measured-round
count, median timing/memory, output state, and source artifact. Missing or
invalid evidence is listed and makes that reporting job fail.

## A. 128-token answer (4 CPU engines, 2 threads)

| Engine | Load ms | First token ms | Speed tok/s | Process wall ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Synapse | 86 | 424 | 104.5 | 1,819 | 570 |
| llama.cpp | n/a | n/a (prompt 170) | 120.3 | 1,928 | 1,251 |
| LLamaSharp | 745 | 137 | 115.5 | 2,108 | 1,272 |
| dotLLM | 398 | 8,122 | 9.8 | 21,823 | 1,290 |

llama.cpp's CLI reports only its internal prompt and generation times. The
longer answers differ between engines after the first sentence and have not
been quality-reviewed.

## B. 3-turn dialogue (4 CPU engines, 2 threads)

Each turn starts a fresh process with the whole transcript, so there is no
KV-cache reuse.

| Turn · prompt tokens | Synapse first token / speed | LLamaSharp first token / speed | llama.cpp prompt / speed | dotLLM first token / speed |
|---|---:|---:|---:|---:|
| 1 · 70 | 425 ms / 74 tok/s | 133 ms / 114 tok/s | 161 ms / 140 tok/s | 8,153 ms / 9.7 tok/s |
| 2 · 165 | 841 ms / 100 tok/s | 287 ms / 87 tok/s | 345 ms / 129 tok/s | 17,849 ms / 9.4 tok/s |
| 3 · 274 | 1,391 ms / 95 tok/s | 473 ms / 105 tok/s | 658 ms / 108 tok/s | 29,592 ms / 9.6 tok/s |

Peak RSS in turn 3: Synapse 574 MiB, LLamaSharp 1,274 MiB, llama.cpp 1,263 MiB,
dotLLM 1,545 MiB. In turn 3, LLamaSharp stopped at 58 tokens and llama.cpp at
61; the others wrote 64.

## C. MLX on the GPU (SwiftLM `b795`, resident server)

| Request | Prompt / output tokens | First token ms | Speed tok/s | Request wall ms | Peak RSS MiB | Prompt tokens served from cache |
|---|---:|---:|---:|---:|---:|---:|
| Answer, 128 tokens | 78 / 128 | 13.8 | 225.5 | 576.5 | 655.6 | 78 |
| Dialogue turn 1 | 77 / 64 | 14.2 | 226.5 | 292.2 | 651.9 | 77 |
| Dialogue turn 2 | 178 / 64 | 40.5 | 225.1 | 320.4 | 651.9 | 77 |
| Dialogue turn 3 | 293 / 64 | 42.3 | 222.7 | 325.6 | 651.9 | 178 |

- The first cold request took 91 ms to the first token. Later requests reused
  the server's prompt cache.
- Request wall time excludes the one-time model load (629–657 ms).
- MLX uses its own 8-bit weights
  ([Qwen2.5 0.5B MLX 8-bit](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-8bit))
  and its own chat template, so prompt token counts differ from the CPU runs.
- Archive SHA-256
  `2ed6b5539b24c5267931d46ea9973775b7d2a9b5ee2f82109afab60f9603675e`,
  weights SHA-256
  `3dd0b6c2983ac5fe35f60ba260b1c7c35e4c38f17f3d5139d0bb477924e7aef4`.
  No Swift build or Python was used.

## D. 8-token smoke test (first scalar Synapse, 12 threads)

Prompt `The capital of France is`, 512-token context, median of 5 runs.

| Engine | Load ms | First token ms | Generation ms | Speed tok/s | Process wall ms | Avg CPU cores | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synapse (scalar) | **56.1** | 328.3 | 633.3 | 24.30 | **691.3** | 10.23 | **559.8** |
| dotLLM | 345.5 | 439.6 | 875.6 | 18.22 | 1,352.5 | 5.66 | 1,190.5 |
| LLamaSharp | 691.2 | **17.2** | **69.0** | **136.16** | 780.9 | 1.76 | 1,137.2 |

All engines produced the same 8 tokens, ` Paris. It is the largest city in`
(Synapse IDs `12095, 13, 1084, 374, 279, 7772, 3283, 304`). Synapse's first
token varied from 304.8 to 574.2 ms. Energy was not measured:
`powermetrics` needs superuser access.

## E. Memory run (first scalar Synapse, 8 threads)

Median of 5 runs, 4 engines rotated each round.

| Engine | Peak RSS MiB | Physical footprint MiB | .NET live heap MiB | Process wall ms |
|---|---:|---:|---:|---:|
| Synapse (scalar) | 560.8 | 39.9 | 20.4 | 895.1 |
| dotLLM | 1,187.3 | 663.1 | not measured | 1,350.9 |
| LLamaSharp | 1,274.9 | 606.9 | 1.1 | 962.6 |
| llama.cpp | 1,258.6 | 594.6 | n/a | 681.6 |

RSS and physical footprint are two different macOS views of the same process
and must not be added. Synapse maps the model file, so its RSS is far above
its footprint. The .NET heap is only part of total memory.

## F. GitHub Actions (first scalar Synapse, 2 threads)

Median of 5 runs. Each operating system is a separate hardware group, not a
leaderboard. Runners: `macos-15` Apple M1, 3 vCPU, 7 GB; `ubuntu-24.04` and
`windows-2025` x64, 4 vCPU, 16 GB.

Writing speed, tokens/s:

| Runner | Synapse | llama.cpp | LLamaSharp | dotLLM |
|---|---:|---:|---:|---:|
| macOS 15 ARM64 | 5.2 | 90.1 | 76.8 | 6.5 |
| Ubuntu 24.04 x64 | 4.6 | 54.8 | 39.4 | 16.6 |
| Windows Server 2025 x64 | 4.0 | 45.5 | 33.8 | 14.7 |

Process wall and peak RSS:

| Runner | Synapse | llama.cpp | LLamaSharp | dotLLM |
|---|---:|---:|---:|---:|
| macOS 15 ARM64 | 2,543 ms / 555 MiB | 978 ms / 1,195 MiB | 1,046 ms / 1,263 MiB | 2,880 ms / 1,166 MiB |
| Ubuntu 24.04 x64 | 2,815 ms / 565 MiB | 535 ms / 725 MiB | 646 ms / 760 MiB | 1,621 ms / 1,174 MiB |
| Windows Server 2025 x64 | 3,502 ms / 551 MiB | 1,014 ms / 575 MiB | 1,092 ms / 596 MiB | 2,424 ms / 1,153 MiB |

All four engines produced the same 8 tokens on every runner.

## G. llama.cpp alone (`b29c606e2`, 12 threads)

- 8 tokens: median process wall 566.2 ms, peak RSS 1,208.3 MiB, internal
  prompt 14.9 ms, generation 57.9 ms (120.84 tok/s).
- 128 tokens: 47.28 to 99.40 tok/s over 5 runs (median 80.73). The Mac was
  busy with other work, so the spread is large.
- `llama-bench`: 115.36 ± 5.94 tok/s. It skips tokenization and sampling, so
  it measures kernels, not a full request.

## H. 32-token output check

Status `ineligible_quality_mismatch`: dotLLM's answer diverged from LLamaSharp
and llama.cpp after a shared start. Synapse has no tokenizer yet, so its
32-token text cannot be compared. These timings are not a speed result.

## How to run

The runner is a C# tool in
[`experiments/Synapse.ReferenceBenchmarks`](../experiments/Synapse.ReferenceBenchmarks).
The [performance workflow](https://github.com/managedcode/Synapse/actions/workflows/performance.yml)
shows the exact commands and pinned engine versions.
