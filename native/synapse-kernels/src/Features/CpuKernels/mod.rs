//! GGML `Q8_0` weight times `Q8_0` activation matrix kernels.

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[allow(unsafe_code)]
mod ffi;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
mod q8_0;
mod scalar;

pub use ffi::{
    ABI_VERSION, STATUS_INVALID_SHAPE, STATUS_NULL_POINTER, STATUS_OK, STATUS_OVERFLOW,
    STATUS_PANIC, synapse_kernels_abi_version, synapse_kernels_capabilities, synapse_q8_0_matmul,
};
pub use q8_0::{
    BLOCK_BYTES, BLOCK_ELEMENTS, Isa, KernelError, LONG_ROW_BLOCKS, Q8Activations, Q8Weights,
    fp16_to_f32, matmul, matmul_with,
};
