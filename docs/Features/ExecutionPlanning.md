# Execution planning

## Implemented behavior

`HeterogeneousPlacementPlanner` splits ordered layer regions into contiguous
stages across one to four devices that differ in memory, weight-streaming
rate, dispatch overhead, and supported encodings.

- It tries every device order and runs a dynamic program over split points.
  This is the bounded search of spec §10.3.
- Each stage reserves KV for the requested context. It then gets the highest
  importance-aware precision that fits the remaining memory and the device's
  kernels (`PrecisionBudgetSelector`).
- `Latency` adds stage and link times for one sequence. `Throughput` takes the
  slowest stage or link, for pipelines serving many sessions.
- Ties prefer lower distortion, then the earlier device order.
- If nothing fits, it raises `PlacementException` with `InsufficientMemory`.
  If a fit exists only with approximation that the request forbids, the
  failure is `ApproximationNotAllowed`.

All costs are planning estimates from supplied profiles. Measured device and
link profiles (TASK-PLN-001) and Orleans integration are still planned; see
`elastic-inference.plan.md` E8.

## Acceptance mapping

| Scenario | Test |
|---|---|
| AC-PLN-002-1 infeasible memory, including KV | `KvReservationCounted` |
| AC-PLN-002-2 unqualified quality plan rejected | `UnqualifiedQualityPlanRejected` |
| AC-PLN-002-3 transfer cost changes the choice | `SlowLinkKeepsModelOnOneDevice` |
| Throughput vs latency objectives | `ThroughputObjectiveGivesFasterDeviceMoreLayers`, `LatencyObjectiveKeepsSequenceOnFastestDevice` |
| Importance-aware precision on a small fast device | `SmallFastDeviceDemotesLeastImportantTensorsWhenAllowed` |
| Bounded search | `SearchIsBoundedToFourDevices` |
