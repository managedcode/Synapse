# Toolchain and dependency lock

Observed on `H-APPLE-DEV` on 2026-09-27:

- macOS 27.0 build 26A428, arm64.
- .NET SDK 10.0.401; runtime 10.0.12.
- Rust 1.98.1 (`48a229cea`, 2026-09-01); Cargo 1.98.1.
- Aspire CLI is installed; the exact version is verified when the first real
  AppHost slice begins. The package line selected from current official
  releases is 13.5.4, not the older 13.4 design-pack observation.
- ZoneTree 1.9.8 and TUnit 1.70.1 are centrally pinned.

The primary development and qualification target is Apple Silicon macOS
(`osx-arm64`). GitHub Actions uses the standard `macos-15` ARM64 runner for
portable CPU/build verification. Actual Metal performance and device evidence
must run on a Metal-capable local or labeled self-hosted Apple runner; a hosted
build without device evidence cannot qualify Metal performance.

`global.json`, `rust-toolchain.toml`, NuGet lock files, and `native/Cargo.lock`
are canonical. Restores use locked mode after the first lock is generated.
Python and Node.js processes or dependencies are forbidden throughout the
product and verification graph.
