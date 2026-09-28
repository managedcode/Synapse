use std::panic::{AssertUnwindSafe, catch_unwind};

use super::q8_0::{
    BLOCK_BYTES, BLOCK_ELEMENTS, Isa, KernelError, Q8Activations, Q8Weights, matmul,
};

/// C ABI version checked by the managed loader before any other call.
pub const ABI_VERSION: u32 = 1;
/// The call completed.
pub const STATUS_OK: i32 = 0;
/// A required pointer was null.
pub const STATUS_NULL_POINTER: i32 = 1;
/// A dimension, stride, or row byte count was inconsistent.
pub const STATUS_INVALID_SHAPE: i32 = 2;
/// A declared size overflowed `usize`.
pub const STATUS_OVERFLOW: i32 = 3;
/// A Rust panic was caught before it reached the caller.
pub const STATUS_PANIC: i32 = 4;

/// Returns [`ABI_VERSION`].
#[unsafe(no_mangle)]
pub const extern "C" fn synapse_kernels_abi_version() -> u32 {
    ABI_VERSION
}

/// Returns the capability bit of the instruction set selected on this CPU.
#[unsafe(no_mangle)]
pub extern "C" fn synapse_kernels_capabilities() -> u32 {
    Isa::detect().capability_bit()
}

/// `Q8_0` matrix kernel: `output[t * output_stride + r] = dot(W[r], A[t])`.
///
/// `weights` points at `row_count` rows of `row_bytes` bytes; activations hold `token_count`
/// rows of `blocks_per_row * 32` codes and `blocks_per_row` scales; `output` holds
/// `(token_count - 1) * output_stride + row_count` floats. Returns a `STATUS_*` code.
///
/// # Safety
///
/// Every non-null pointer must be valid for the lengths above for the whole call, and
/// `output` must not overlap the inputs.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn synapse_q8_0_matmul(
    weights: *const u8,
    row_bytes: usize,
    row_count: usize,
    blocks_per_row: usize,
    activation_quants: *const i8,
    activation_scales: *const f32,
    token_count: usize,
    output: *mut f32,
    output_stride: usize,
) -> i32 {
    if weights.is_null()
        || activation_quants.is_null()
        || activation_scales.is_null()
        || output.is_null()
    {
        return STATUS_NULL_POINTER;
    }

    if row_count == 0
        || blocks_per_row == 0
        || token_count == 0
        || blocks_per_row.checked_mul(BLOCK_BYTES) != Some(row_bytes)
    {
        return STATUS_INVALID_SHAPE;
    }

    let Some(lengths) = Lengths::new(
        row_bytes,
        row_count,
        blocks_per_row,
        token_count,
        output_stride,
    ) else {
        return STATUS_OVERFLOW;
    };

    // SAFETY: the pointers are non-null and the caller guarantees they are valid for the
    // lengths computed from the same shape; the output does not alias the inputs.
    let (weights, quants, scales, output) = unsafe {
        (
            std::slice::from_raw_parts(weights, lengths.weights),
            std::slice::from_raw_parts(activation_quants, lengths.codes),
            std::slice::from_raw_parts(activation_scales, lengths.scales),
            std::slice::from_raw_parts_mut(output, lengths.output),
        )
    };
    let weights = Q8Weights {
        data: weights,
        rows: row_count,
        blocks_per_row,
    };
    let activations = Q8Activations {
        quants,
        scales,
        tokens: token_count,
    };
    match catch_unwind(AssertUnwindSafe(|| {
        matmul(&weights, &activations, output, output_stride)
    })) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(KernelError::Overflow)) => STATUS_OVERFLOW,
        Ok(Err(_)) => STATUS_INVALID_SHAPE,
        Err(_) => STATUS_PANIC,
    }
}

struct Lengths {
    weights: usize,
    codes: usize,
    scales: usize,
    output: usize,
}

impl Lengths {
    fn new(
        row_bytes: usize,
        rows: usize,
        blocks: usize,
        tokens: usize,
        stride: usize,
    ) -> Option<Self> {
        let scales = tokens.checked_mul(blocks)?;
        Some(Self {
            weights: rows.checked_mul(row_bytes)?,
            codes: scales.checked_mul(BLOCK_ELEMENTS)?,
            scales,
            output: (tokens - 1).checked_mul(stride)?.checked_add(rows)?,
        })
    }
}
