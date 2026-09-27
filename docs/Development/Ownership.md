# Ownership

The primary integration owner controls root build files, `Synapse.slnx`,
public contracts, workspace manifests, protocol schemas, ADRs, and package
versions. Feature changes stay within their matching `Features/<SliceName>`
paths. Cross-slice public changes require an ADR and an integrated review.

Initial ownership:

| Area | Owner boundary |
|---|---|
| Bootstrap | Toolchain locks, doctor contracts, process verification |
| GraphExecution | C# verified IR, kernels, model execution; optional profiled native acceleration |
| SessionState | C# hot KV ownership and ZoneTree metadata adapter |
| Benchmarking | External adapters, run schema, raw evidence, verdicts |
| AppHost | Topology only; no inference or domain logic |
