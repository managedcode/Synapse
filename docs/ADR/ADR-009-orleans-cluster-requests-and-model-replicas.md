# ADR-009: Orleans cluster, request grains, and model replica placement

Status: Accepted as the owner's direction. Date: 2026-09-28. Implementation: not
started; this ADR fixes the topology before code, as `AGENTS.md` requires.

## Context

The owner's direction:
- Synapse starts as an Orleans cluster, and nodes join it.
- Every inference request is its own grain, for isolation.
- Loaded weights exist once per node, or once per cluster when one model is
  split across nodes.
- Balancing uses what Orleans 10 already provides.

ADR-007 made one model instance batch many concurrent requests. The
Architecture keeps these constraints:
- Large tensors, activations, and KV never pass through Orleans.
- There is no grain per token, layer, or neuron.
- Local mode runs without Orleans.

Orleans 10 capabilities used here, checked against Microsoft Learn on
2026-09-28:
- `ResourceOptimizedPlacement` has been the default since 9.2, and custom
  `IPlacementDirector` implementations are supported.
- Grain placement filtering can use silo metadata (`RequiredMatch` and
  `PreferredMatch`).
- `ActivationRebalancer` (experimental, `ORLEANSEXP002`) and the activation
  repartitioner (`ORLEANSEXP001`) move activations automatically.
  `[Immovable]` excludes a grain from both.
- Grain methods can stream `IAsyncEnumerable<T>` with batching and
  backpressure, and fully support `CancellationToken`.
- Memory-based activation shedding, the strong-consistency grain directory,
  load shedding via `CpuThreshold`, and the Orleans Dashboard.

Orleans hosting is not a NativeAOT target, so NativeAOT is rejected for silo
hosts (ADR-006 CPU.4).

## Decision

### Roles

| Piece | Kind | Key | Owns |
|---|---|---|---|
| `NodeModelHost` | DI singleton per silo (not a grain) | none | Resident model replicas on this node: mmap weights, worker pool, `ContinuousBatchScheduler`, KV slots |
| `InferenceRequestGrain` | Grain, `[Immovable]` | Request ID (GUID) | One request's validated input, cancellation, fenced epoch, streamed output, final status |
| `ModelDeploymentGrain` | Grain | Model fingerprint | Replication mode, replica set, desired replica count, memory budget per node |
| `ModelResidencyGrain` | Grain | Model fingerprint | Live replica load: free KV slots, queue depth, measured tok/s, prefix-cache summaries, reported about once per second |
| `ConversationGrain` (optional) | Grain | Conversation ID | Prefix affinity: which node holds a conversation's reusable KV |

- **Request isolation.** A request grain never shares mutable state with
  another request. Its KV slot lives in the chosen node's scheduler and
  belongs to it alone. When the grain completes or is canceled, the slot is
  released.
- **Per-node calls stay in process.** A request grain always runs on a silo
  with a resident replica. It calls `NodeModelHost.GenerateAsync` through
  dependency injection, not through a grain call. It streams token IDs back to
  the client as `IAsyncEnumerable<GeneratedTokenChunk>`, which Orleans
  batches. Steady-state decode makes no grain-to-grain calls.

### Weights: three replication modes, chosen per model

1. **`PerNode`** (the default when a model fits a node). Every eligible node
   maps its own copy of the weights. The mmap is shared by all requests on
   that node, and nothing crosses nodes per token. Capacity scales with nodes.
2. **`Subset(k)`**. Only `k` nodes hold replicas. `ModelDeploymentGrain` picks
   them from memory budgets and demand, and changes `k` at safe points with
   hysteresis. Other nodes host other models.
3. **`PerCluster`**. One model is split across nodes (a layer-range pipeline,
   `flybrain.plan.md` F6.1, or head/position shards, F6.4). Use it only when
   the weights or KV do not fit a node. The request grain is placed on the
   stage-0 node. Stages exchange batch step packets over the direct data plane
   (ADR-007 step plus `session, epoch, step, position, plan hash`), never
   through Orleans. The existing heterogeneous planner picks the split.

### Balancing, from coarse to fine

1. **Capability filter.** Silos declare static metadata at start, for example
   `synapse.cpu=arm64-sdot` or `x64-avx2`, `synapse.memory-class`, and
   `synapse.zone`. `RequiredMatch` excludes silos that cannot run the model's
   encoding or kernel. `PreferredMatch` keeps requests in the caller's zone.
2. **Request placement.** A custom `ModelReplicaPlacement` director runs
   after the filters. It reads the cached `ModelResidencyGrain` snapshot and
   picks, in order:
   1. a replica that already holds the request's prefix (from the
      conversation or the ZoneTree prefix index);
   2. the replica with the best score, where
      `score = free KV slots × measured tok/s ÷ (queue depth + 1)`;
   3. failing both, `ResourceOptimizedPlacement` among silos with a replica.

   If no replica has capacity, it asks `ModelDeploymentGrain` to scale out,
   or it queues the request with a visible wait. It never overcommits KV.
3. **Node-local balancing.** ADR-007 continuous batching. Decode goes first,
   prompts are chunked, and admission is bounded.
4. **Cluster rebalancing.** New requests move, not running ones. Request
   grains are `[Immovable]`, because migration would strand their hot KV.
   `ActivationRebalancer` may move movable control grains.
   `ModelDeploymentGrain` adds or removes replicas to follow demand.
5. **Overload.** `LoadSheddingOptions.CpuThreshold` and memory-based shedding
   reject new work explicitly. A shed or deactivated request grain cancels its
   generation and returns a typed error. It never truncates output silently.

### Failure and consistency

- A node loss drops its request grains and their hot KV. Clients see a typed
  failure and retry as a new request.
- Pipeline stages reject a stale epoch or plan hash at the receiver.
- Exactly-once delivery is not claimed.

## Consequences

- New projects: an Orleans 10 silo host with an Aspire AppHost for local
  multi-node runs. The runtime stays Orleans-free, so local mode remains
  unchanged.
- The first tests use the in-process test cluster with two or more silos:
  - `RequestTokensEqualLocalRun`
  - `RequestsPlacedOnlyOnResidentReplicas`
  - `CancelledRequestReleasesSlot`
  - `NoGrainCallsInSteadyStateDecode` (counted by a grain call filter)
  - `PerNodeReplicaSharedByRequests`
- The implementation order:
  1. single-model `PerNode`, then residency-aware placement;
  2. `ModelDeploymentGrain` scaling;
  3. `PerCluster` pipelines through F6.1;
  4. conversation prefix affinity with ZoneTree.
- Two measurements come before any capacity claim: requests per second and
  p50/p95 latency against node count.
