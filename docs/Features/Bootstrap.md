# Bootstrap

## Contract

`REQ-BST-002` pins observed toolchains and dependencies. `REQ-BST-003`
provides a genuine offline doctor in both C# and Rust.

The doctor accepts a positive memory budget in bytes, reports architecture and
available CPU vector capabilities, and rejects invalid budgets with
`InvalidMemoryBudget`. When a state directory is supplied, the C# doctor also
executes a durable ZoneTree write, close, reopen, and read.

## Acceptance mapping

| Acceptance | Test | Evidence expectation |
|---|---|---|
| AC-BST-002-1 | TEST-BST-002-1 | Locked NuGet/Cargo restore succeeds twice. |
| AC-BST-002-2 | TEST-BST-002-2 | Policy audit rejects Python/Node dependencies. |
| AC-BST-002-3 | TEST-BST-002-3 | SDK mismatch is explained before build. |
| AC-BST-003-1 | TEST-BST-003-1 | C# and Rust doctors report real CPU facts offline. |
| AC-BST-003-2 | TEST-BST-003-2 | Zero/negative budgets return a typed failure. |
| AC-BST-003-3 | TEST-BST-003-3 | Workspace builds and both suites run non-zero tests. |

## Non-goals

This feature is limited to bootstrap behavior. Managed Qwen2 model execution
is tracked separately; neither feature currently claims Metal qualification,
statistical performance results, or distributed readiness.
