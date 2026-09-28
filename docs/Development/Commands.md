# Commands

Run from the repository root. Add `/opt/homebrew/opt/rustup/bin` to `PATH` on
this Apple machine because Homebrew installs rustup keg-only.

```text
dotnet restore Synapse.slnx --locked-mode
dotnet build Synapse.slnx --configuration Release --no-restore
dotnet test Synapse.slnx --configuration Release --no-build

cargo test --manifest-path native/Cargo.toml --locked
cargo fmt --manifest-path native/Cargo.toml --all --check
cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings

dotnet run --project src/Synapse.Cli -- doctor --memory-budget-bytes 1073741824
dotnet run --project src/Synapse.Cli -- model fetch --set smoke
dotnet run --project src/Synapse.Cli --configuration Release -- generate \
  --model artifacts/models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --tokens 785,6722,315,9625,374 --max-tokens 8 --context-size 512 --threads 12
cargo run --manifest-path native/Cargo.toml --locked -p synapse-runtime -- \
  doctor --memory-budget-bytes 1073741824
```

No command in this repository invokes Python or Node.js.

GPU and long-context runs (ADR-012, ADR-013). `--backend metal` needs an Apple7+
GPU; `--backend cuda` needs an NVIDIA driver and NVRTC at runtime. Contexts above
the trained window need an explicit `--rope-scaling yarn:<factor>:<trained>`.
`SYNAPSE_GPU_TRACE=1` prints the GPU milliseconds of every step to standard error.

```text
dotnet run --project src/Synapse.Cli --configuration Release -- generate \
  --model artifacts/models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --tokens-file prompt-tokens.txt --max-tokens 64 --context-size 131072 \
  --backend metal --rope-scaling yarn:4:32768
dotnet run --project experiments/Synapse.ReferenceBenchmarks --configuration Release -- passkey \
  --synapse-executable src/Synapse.Cli/bin/Release/net10.0/synapse \
  --model artifacts/models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf \
  --scenario benchmarks/scenarios/passkey-qwen2.5.json --prompt-tokens 4096,16384 \
  --depths 0.1,0.5,0.9 --backend metal --context-size 32768 --output passkey.json
```

Tokenizer and quality evaluation (ADR-014, ADR-015). The pinned corpora are
built from a commit with `git show`, so they do not depend on the working tree.
llama.cpp tools need `--no-escape`, because they otherwise rewrite `\n`-style
escapes. llama.cpp `-f` also drops one trailing newline from a prompt file.

```text
synapse tokenize --model <model.gguf> --text-file <text> [--no-parse-special]
synapse detokenize --model <model.gguf> --tokens-file <ids> [--special]
git ls-tree -r --name-only b090e95 | grep -E '\.(md|cs|rs)$' | sort \
  | while read f; do git show "b090e95:$f"; printf '\n'; done > corpus-all.txt
synapse score --model <model.gguf> --text-file corpus-all.txt --context-size 32768 \
  --chunks 7 --backend metal --kv-precision f16 --scores-output trace.json
llama-perplexity -m <model.gguf> -f corpus-all.txt -c 32768 --chunks 7 -ngl 99 -fa on --no-escape
dotnet run --project experiments/Synapse.ReferenceBenchmarks --configuration Release -- quality \
  --model <model.gguf> --haystack haystack.txt --lengths 4000,8000,16000,32000 \
  --depths 0.1,0.5,0.9 --context-size 32768 --output quality.json \
  --subjects synapse:metal:f32,synapse:metal:f16,llamacpp:metal,mlx \
  --synapse-executable src/Synapse.Cli/bin/Release/net10.0/synapse \
  --mlx-binary <SwiftLM> --mlx-model <mlx-qwen2.5-0.5b-8bit-directory>
```
