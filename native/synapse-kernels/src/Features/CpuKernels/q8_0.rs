use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

/// Values per GGML `Q8_0` block.
pub const BLOCK_ELEMENTS: usize = 32;
/// Bytes per GGML `Q8_0` block: an FP16 scale followed by 32 signed codes.
pub const BLOCK_BYTES: usize = 34;
/// Rows with at least this many blocks stream one row at a time (measured faster).
pub const LONG_ROW_BLOCKS: usize = 64;

/// Shape or buffer errors detected before any kernel reads memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    /// A dimension is zero or the stride cannot hold one token's rows.
    InvalidShape,
    /// A buffer is shorter than the declared shape requires.
    BufferTooSmall,
    /// A declared size overflows `usize`.
    Overflow,
    /// The requested instruction set is not available on this CPU.
    UnsupportedIsa,
}

impl Display for KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidShape => "Q8_0 kernel shape is invalid",
            Self::BufferTooSmall => "Q8_0 kernel buffer is shorter than its shape",
            Self::Overflow => "Q8_0 kernel size overflows",
            Self::UnsupportedIsa => "Q8_0 kernel instruction set is unavailable",
        })
    }
}

impl std::error::Error for KernelError {}

/// Kernel instruction-set variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Isa {
    /// Portable integer loops.
    Scalar,
    /// `AArch64` NEON with the `sdot` dot-product extension.
    NeonDotProd,
    /// `x86_64` AVX2 with FMA.
    Avx2,
}

impl Isa {
    /// Returns the fastest instruction set available on this CPU.
    #[must_use]
    pub fn detect() -> Self {
        static DETECTED: OnceLock<Isa> = OnceLock::new();
        *DETECTED.get_or_init(|| {
            [Self::NeonDotProd, Self::Avx2]
                .into_iter()
                .find(|isa| isa.is_supported())
                .unwrap_or(Self::Scalar)
        })
    }

    /// Reports whether this instruction set can run on the current CPU.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::NeonDotProd => neon_supported(),
            Self::Avx2 => avx2_supported(),
        }
    }

    /// Capability bit reported through the C ABI.
    #[must_use]
    pub const fn capability_bit(self) -> u32 {
        match self {
            Self::Scalar => 1,
            Self::NeonDotProd => 1 << 1,
            Self::Avx2 => 1 << 2,
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_supported() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
        && std::arch::is_aarch64_feature_detected!("dotprod")
}

#[cfg(not(target_arch = "aarch64"))]
const fn neon_supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx2_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
const fn avx2_supported() -> bool {
    false
}

/// Row-major `Q8_0` weights for `rows` rows of `blocks_per_row` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q8Weights<'a> {
    /// Encoded blocks, at least `rows * blocks_per_row * 34` bytes.
    pub data: &'a [u8],
    /// Output rows computed by the call.
    pub rows: usize,
    /// Blocks in each row.
    pub blocks_per_row: usize,
}

/// `Q8_0` activation rows for `tokens` tokens with the same width as the weights.
#[derive(Clone, Copy, Debug)]
pub struct Q8Activations<'a> {
    /// Signed codes, `blocks_per_row * 32` per token.
    pub quants: &'a [i8],
    /// Block scales, `blocks_per_row` per token.
    pub scales: &'a [f32],
    /// Token rows to multiply.
    pub tokens: usize,
}

/// Exact FP16 to FP32 conversion for finite values (magnitude shift plus 2^112 rescale).
#[must_use]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let magnitude = f32::from_bits(u32::from(bits & 0x7fff) << 13) * f32::from_bits(0x7780_0000);
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Computes `output[t * stride + r] = dot(W[r], A[t])` with the fastest available ISA.
///
/// # Errors
///
/// Returns a [`KernelError`] when a dimension is zero, a buffer is too short, or a size
/// overflows; no memory is read in that case.
pub fn matmul(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    output_stride: usize,
) -> Result<(), KernelError> {
    matmul_with(Isa::detect(), weights, activations, output, output_stride)
}

/// Same as [`matmul`] with an explicit instruction set.
///
/// # Errors
///
/// Returns [`KernelError::UnsupportedIsa`] when `isa` cannot run here, or the shape errors
/// described for [`matmul`].
pub fn matmul_with(
    isa: Isa,
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    output_stride: usize,
) -> Result<(), KernelError> {
    validate(weights, activations, output.len(), output_stride)?;
    if !isa.is_supported() {
        return Err(KernelError::UnsupportedIsa);
    }

    match isa {
        Isa::Scalar => {
            crate::cpu_kernels::scalar::matmul(weights, activations, output, output_stride);
        }
        Isa::NeonDotProd => run_neon(weights, activations, output, output_stride),
        Isa::Avx2 => run_avx2(weights, activations, output, output_stride),
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn run_neon(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    stride: usize,
) {
    // SAFETY: `validate` proved every buffer covers the declared shape and `matmul_with`
    // proved the CPU supports NEON dotprod before dispatching here.
    unsafe { crate::cpu_kernels::neon::matmul(weights, activations, output, stride) }
}

#[cfg(not(target_arch = "aarch64"))]
const fn run_neon(_: &Q8Weights<'_>, _: &Q8Activations<'_>, _: &mut [f32], _: usize) {}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn run_avx2(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    stride: usize,
) {
    // SAFETY: `validate` proved every buffer covers the declared shape and `matmul_with`
    // proved the CPU supports AVX2 and FMA before dispatching here.
    unsafe { crate::cpu_kernels::avx2::matmul(weights, activations, output, stride) }
}

#[cfg(not(target_arch = "x86_64"))]
const fn run_avx2(_: &Q8Weights<'_>, _: &Q8Activations<'_>, _: &mut [f32], _: usize) {}

fn validate(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output_length: usize,
    output_stride: usize,
) -> Result<(), KernelError> {
    let rows = weights.rows;
    let blocks = weights.blocks_per_row;
    let tokens = activations.tokens;
    if rows == 0 || blocks == 0 || tokens == 0 || (tokens > 1 && output_stride < rows) {
        return Err(KernelError::InvalidShape);
    }

    let weight_bytes = rows
        .checked_mul(blocks)
        .and_then(|count| count.checked_mul(BLOCK_BYTES))
        .ok_or(KernelError::Overflow)?;
    let scale_count = tokens.checked_mul(blocks).ok_or(KernelError::Overflow)?;
    let code_count = scale_count
        .checked_mul(BLOCK_ELEMENTS)
        .ok_or(KernelError::Overflow)?;
    let output_count = (tokens - 1)
        .checked_mul(output_stride)
        .and_then(|count| count.checked_add(rows))
        .ok_or(KernelError::Overflow)?;
    if weights.data.len() < weight_bytes
        || activations.quants.len() < code_count
        || activations.scales.len() < scale_count
        || output_length < output_count
    {
        return Err(KernelError::BufferTooSmall);
    }

    Ok(())
}
