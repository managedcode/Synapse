# 2026-09-28 longer-workload and MLX diagnostic evidence

Machine: MacBook Pro M2 Pro, macOS 27 arm64, 32 GB unified memory. This is
local diagnostic evidence, not an Actions result or a quality-qualified speed
comparison. The CPU raw files contain one measured round and zero warm-ups;
the MLX files contain one warm-up and three measured requests per turn. CPU
GGUF and MLX 8-bit Metal use different weight and execution cohorts.

| Raw file | Status |
|---|---|
| [CPU 128-token single request](2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-single-128-diagnostic.json) | Measured locally; all four CPU subjects, one round |
| [CPU France/US/UK dialogue](2026-09-28-m2-pro-qwen2.5-0.5b-q8_0-capitals-3turn-diagnostic.json) | Measured locally; fresh process per subject and turn, no retained KV |
| [MLX 128-token single request](2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-single-128-diagnostic.json) | Measured locally; resident prebuilt SwiftLM Metal server |
| [MLX France/US/UK dialogue](2026-09-28-m2-pro-mlx-qwen2.5-0.5b-8bit-capitals-3turn-diagnostic.json) | Measured locally; pinned-server log records observed prompt-cache hit tokens |

The CPU run was captured while other development activity continued and uses
an in-progress managed CPU implementation; its exact executable SHA-256 was
not retained in this early diagnostic. The model and scenario hashes, actual
prompt/output token counts, raw subject results, CPU, wall and whole-process
memory are in JSON. The MLX raw evidence also includes binary, Metal library,
model-weight and scenario SHA-256 hashes. No formal paired winner claim is
supported. A manual spot-check found incorrect or truncated location claims
in baseline text, and Synapse's longer token IDs have not been quality-decoded.

Verified external artifact: SwiftLM `b795` release archive, 53,329,610 bytes,
SHA-256 `2ed6b5539b24c5267931d46ea9973775b7d2a9b5ee2f82109afab60f9603675e`.
The local archive was checked with `shasum -a 256`; the binary was executed
from the extracted archive. There was no local Swift build or Python path.
Qwen MLX model revision `6474d4c1a05bd7bd634a9c993560e4e7272eda3d`
was fetched through the C# catalog (`model fetch` exit 0, nine verified files).

Verification commands and observed results:

```text
actionlint .github/workflows/performance.yml
  exit 0
dotnet build Synapse.slnx --configuration Release --no-restore --disable-build-servers -v minimal
  exit 0; 0 warnings, 0 errors
dotnet test tests/Synapse.IntegrationTests/Synapse.IntegrationTests.csproj --configuration Release --treenode-filter '/*/*/DialogueReportTests/*' --progress on
  red exit 2 after adding Avg CPU cores assertion; green exit 0, 2/2 after implementation
dotnet test tests/Synapse.IntegrationTests/Synapse.IntegrationTests.csproj --configuration Release --no-build --treenode-filter '/*/*/DiagnosticMatrixReportTests/*|/*/*/ModelPackageCatalogTests/*' --progress on
  exit 0, 2/2 diagnostic-report cases
dotnet test tests/Synapse.IntegrationTests/Synapse.IntegrationTests.csproj --configuration Release --no-build --treenode-filter '/*/*/ModelPackageCatalogTests/*' --progress on
  exit 0, 6/6 catalog cases
dotnet format Synapse.slnx --verify-no-changes --no-restore --include experiments/Synapse.ReferenceBenchmarks/DiagnosticMatrixReportCommand.cs experiments/Synapse.ReferenceBenchmarks/DialogueReportCommand.cs experiments/Synapse.ReferenceBenchmarks/LockedDialogueCommand.cs experiments/Synapse.ReferenceBenchmarks/MlxBenchmarkCommand.cs experiments/Synapse.ReferenceBenchmarks/Program.cs tests/Synapse.IntegrationTests/Features/Benchmarking/DiagnosticMatrixReportTests.cs tests/Synapse.IntegrationTests/Features/Benchmarking/DialogueReportTests.cs tests/Synapse.IntegrationTests/Features/ModelPackages/ModelPackageCatalogTests.cs
  exit 0
dotnet test Synapse.slnx --configuration Release --no-build --progress on
  exit 2; 151 passed / 158 total, 7 failed because the required external llama.cpp
  and dotLLM executable environment variables were unset and the concurrent
  CPU-kernels branch had no built libsynapse_kernels.dylib in the test output
```

The new GitHub Actions long-workload and MLX jobs have not yet run. Their
workflow is syntax-checked, but remote runner compatibility and result quality
are not proven by this local evidence.
