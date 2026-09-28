//! Synapse GPU execution backends behind a versioned C ABI (ADR-012).
//!
//! The managed runtime owns scheduling, sampling, session ownership, and the memory-mapped weights. This crate
//! owns device buffers, shader pipelines, and command encoding. Unsupported devices, shapes, and backends fail
//! with explicit status codes; nothing falls back silently.

#[path = "Features/GpuKernels/mod.rs"]
pub mod gpu_kernels;
