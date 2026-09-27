# Commands

Run from the repository root. Add `/opt/homebrew/opt/rustup/bin` to `PATH` on
this Apple machine because Homebrew installs rustup keg-only.

```text
dotnet restore Synapse.slnx --use-lock-file
dotnet build Synapse.slnx --configuration Release --locked-mode
dotnet test Synapse.slnx --configuration Release --no-build

cargo test --manifest-path native/Cargo.toml --locked
cargo fmt --manifest-path native/Cargo.toml --all --check
cargo clippy --manifest-path native/Cargo.toml --workspace --all-targets -- -D warnings

dotnet run --project src/Synapse.Cli -- doctor --memory-budget-bytes 1073741824
dotnet run --project src/Synapse.Cli --configuration Release -- generate \
  --model tests/Fixtures/Models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf \
  --tokens 785,6722,315,9625,374 --max-tokens 1 --context-size 512
cargo run --manifest-path native/Cargo.toml --locked -p synapse-runtime -- \
  doctor --memory-budget-bytes 1073741824
```

No command in this repository invokes Python or Node.js.
