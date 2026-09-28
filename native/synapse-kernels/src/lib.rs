//! Optimized Synapse CPU kernels behind a versioned C ABI (ADR-006).
//!
//! The safe [`cpu_kernels::matmul`] API validates every shape before an ISA-specific
//! implementation touches memory. The `extern "C"` surface in [`cpu_kernels`] converts
//! caller-owned buffers into slices, returns status codes, and never lets a panic cross
//! the ABI boundary.

#[path = "Features/CpuKernels/mod.rs"]
pub mod cpu_kernels;
