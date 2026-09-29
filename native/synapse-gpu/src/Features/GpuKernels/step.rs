//! Backend-neutral encoding of one batched decoder step.
//!
//! The step covers embedding, every transformer block, and the logits rows. Metal and CUDA implement
//! [`StepBackend`]; both run the same kernel sequence, arguments, and geometry, so a behavior change lands
//! once.

mod activations;
mod attention;
mod layers;
mod projection;

pub use activations::Activations;

use super::decoder::{DecoderPlan, Encoding, KvPrecision, NO_TENSOR};
use super::error::GpuError;
use super::params::{NO_BIAS, SlotTable};
use super::schedule::AttentionPlan;

/// Query-key pairs per step above which every layer becomes its own device submission. A single long
/// submission (for example 512 prompt tokens at 84k context) trips the macOS GPU watchdog.
const SUBMISSION_PAIRS: u64 = 1 << 22;
const _: () = assert!(NO_BIAS == NO_TENSOR);

/// A kernel of the step. `Matvec(e, t)` processes `t` tokens (1, 2, 4, or 8) per threadgroup of weights in
/// encoding `e` (ADR-021).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Embed(Encoding),
    RmsNorm,
    Matvec(Encoding, u32),
    Gemm(Encoding),
    RopeKv(KvPrecision),
    Attention(KvPrecision),
    AttentionDecode(KvPrecision),
    AttentionReduce,
    Swiglu,
}

/// One kernel parameter after the argument struct at index 0.
pub enum Binding<'a, B> {
    Buffer {
        index: usize,
        buffer: &'a B,
        offset: usize,
    },
    Table {
        index: usize,
        table: &'a SlotTable,
    },
}

impl<B> Binding<'_, B> {
    /// A whole buffer bound at `index`.
    pub const fn whole(index: usize, buffer: &B) -> Binding<'_, B> {
        Binding::Buffer {
            index,
            buffer,
            offset: 0,
        }
    }
}

/// A device that runs the step kernels in submission order.
pub trait StepBackend {
    type Buffer;

    /// Launches `kernel` with `args` at index 0 and `bindings` at their indices.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Device`] when the backend rejects the launch.
    fn launch<T: Copy>(
        &mut self,
        kernel: Kernel,
        args: &T,
        bindings: &[Binding<'_, Self::Buffer>],
        groups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<(), GpuError>;

    /// Weight rows per matrix-vector threadgroup for `tokens_per_group` tokens of `encoding` weights; the grid is
    /// sized from it.
    fn matvec_rows(&self, encoding: Encoding, tokens_per_group: u32) -> u32 {
        let _ = (encoding, tokens_per_group);
        2
    }

    /// Ends the current submission; later launches still run after it.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Device`] when the submission cannot be issued.
    fn flush(&mut self) -> Result<(), GpuError>;
}

/// One mapped weight segment (ADR-019): file bytes `[first, end)` live in `buffer` from byte `buffer_offset`.
pub struct WeightSegment<B> {
    pub first: u64,
    pub end: u64,
    pub buffer: B,
    pub buffer_offset: u64,
}

/// The segment buffer that holds file offset `offset`, and the byte offset inside it.
///
/// # Errors
///
/// Returns [`GpuError::InvalidArgument`] when no segment covers the offset.
pub fn locate<B>(segments: &[WeightSegment<B>], offset: u64) -> Result<(&B, u64), GpuError> {
    segments
        .iter()
        .find(|segment| segment.first <= offset && offset < segment.end)
        .map(|segment| {
            (
                &segment.buffer,
                segment.buffer_offset + (offset - segment.first),
            )
        })
        .ok_or_else(|| {
            GpuError::invalid(format!(
                "weight offset {offset} lies outside every mapped segment"
            ))
        })
}

/// Everything one step reads besides the token list.
pub struct StepInputs<'a, B> {
    pub plan: &'a DecoderPlan,
    /// The mapped weights: one segment, or several when layers are dropped (ADR-019).
    pub weights: &'a [WeightSegment<B>],
    pub cosines: &'a B,
    pub sines: &'a B,
    pub activations: &'a Activations<B>,
    pub table: &'a SlotTable,
    pub attention: &'a AttentionPlan,
}

/// Encodes one validated step of `tokens` tokens with `logits_rows` logits rows.
///
/// # Errors
///
/// Propagates backend launch and submission failures.
pub fn encode_step<S: StepBackend>(
    backend: &mut S,
    inputs: &StepInputs<'_, S::Buffer>,
    tokens: usize,
    logits_rows: u32,
) -> Result<(), GpuError> {
    let count = u32::try_from(tokens).unwrap_or(u32::MAX);
    let attention = inputs.attention;
    // Tokens before the first prompt run are decode tokens (the scheduler orders them first).
    let gemm_from = attention.blocks[..attention.run_blocks as usize]
        .iter()
        .map(|block| block.first_token)
        .min()
        .unwrap_or(count);
    let longest = attention
        .blocks
        .iter()
        .map(|block| block.key_end)
        .max()
        .unwrap_or(0);
    let split = u64::from(count) * u64::from(longest) >= SUBMISSION_PAIRS;
    let mut step = layers::Step {
        backend,
        inputs,
        span: (count, gemm_from),
    };
    step.embed()?;
    let layers = inputs.plan.layers.len();
    for (layer, offsets) in inputs.plan.layers.iter().enumerate() {
        let index = u32::try_from(layer).unwrap_or(u32::MAX);
        step.attention_half(index, offsets)?;
        step.feed_forward_half(offsets)?;
        if split && layer + 1 < layers {
            step.backend.flush()?;
        }
    }

    if logits_rows > 0 {
        step.logits(logits_rows)?;
    }

    Ok(())
}
