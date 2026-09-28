//! Step activation buffers and their sizes, shared by every backend.

use crate::gpu_kernels::decoder::DecoderPlan;
use crate::gpu_kernels::schedule::{PARTIAL_BLOCKS, PARTIAL_ROW_FLOATS};

/// Step activations sized for the model's step capacity.
pub struct Activations<B> {
    pub tokens: B,
    pub hidden: B,
    pub normalized: B,
    pub qkv: B,
    pub attention: B,
    pub gate_up: B,
    pub feed_forward: B,
    pub logits_input: B,
    pub logits: B,
    pub gather: B,
    pub blocks: B,
    pub partial: B,
}

impl Activations<u64> {
    /// Byte sizes of every step buffer for `plan`.
    #[must_use]
    pub fn sizes(plan: &DecoderPlan) -> Self {
        let shape = &plan.shape;
        let step = u64::from(shape.step_tokens);
        let floats = |count: u64| count * 4;
        let blocks = (step * u64::from(shape.kv_heads)).max(1);
        Self {
            tokens: step * 16,
            hidden: floats(step * u64::from(shape.hidden)),
            normalized: floats(step * u64::from(shape.hidden)),
            qkv: floats(step * u64::from(shape.qkv_width())),
            attention: floats(step * u64::from(shape.hidden)),
            gate_up: floats(step * 2 * u64::from(shape.feed_forward)),
            feed_forward: floats(step * u64::from(shape.feed_forward)),
            logits_input: floats(u64::from(shape.logits_rows) * u64::from(shape.hidden)),
            logits: floats(u64::from(shape.logits_rows) * u64::from(shape.vocabulary)),
            gather: 4 * u64::from(shape.logits_rows),
            blocks: blocks * 32,
            partial: floats(u64::from(PARTIAL_BLOCKS) * 64 * u64::from(PARTIAL_ROW_FLOATS)),
        }
    }
}

impl<T> Activations<T> {
    /// Converts every buffer description, stopping at the first failure.
    ///
    /// # Errors
    ///
    /// Returns the first error of `convert`.
    pub fn try_map<U, E>(
        self,
        mut convert: impl FnMut(&'static str, T) -> Result<U, E>,
    ) -> Result<Activations<U>, E> {
        Ok(Activations {
            tokens: convert("tokens", self.tokens)?,
            hidden: convert("hidden", self.hidden)?,
            normalized: convert("normalized", self.normalized)?,
            qkv: convert("qkv", self.qkv)?,
            attention: convert("attention", self.attention)?,
            gate_up: convert("gate_up", self.gate_up)?,
            feed_forward: convert("feed_forward", self.feed_forward)?,
            logits_input: convert("logits input", self.logits_input)?,
            logits: convert("logits", self.logits)?,
            gather: convert("gather", self.gather)?,
            blocks: convert("attention blocks", self.blocks)?,
            partial: convert("attention partial", self.partial)?,
        })
    }
}
