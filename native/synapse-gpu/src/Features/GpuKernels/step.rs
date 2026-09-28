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

use super::decoder::{DecoderPlan, KvPrecision, NO_TENSOR};
use super::error::GpuError;
use super::params::{NO_BIAS, SlotTable};
use super::schedule::AttentionPlan;

/// Query-key pairs per step above which every layer becomes its own device submission. A single long
/// submission (for example 512 prompt tokens at 84k context) trips the macOS GPU watchdog.
const SUBMISSION_PAIRS: u64 = 1 << 22;
const _: () = assert!(NO_BIAS == NO_TENSOR);

/// A kernel of the step. `Matvec(t)` processes `t` tokens (1, 2, 4, or 8) per threadgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Embed,
    RmsNorm,
    Matvec(u32),
    Gemm,
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

    /// Ends the current submission; later launches still run after it.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Device`] when the submission cannot be issued.
    fn flush(&mut self) -> Result<(), GpuError>;
}

/// Everything one step reads besides the token list.
pub struct StepInputs<'a, B> {
    pub plan: &'a DecoderPlan,
    pub weights: &'a B,
    pub weights_offset: u64,
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
