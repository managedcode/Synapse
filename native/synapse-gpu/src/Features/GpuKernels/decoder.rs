//! Architecture-neutral dense decoder description shared by every GPU backend (ADR-012).
//!
//! A family adapter (Qwen2, Llama, Mistral, ...) maps its tensors onto this pre-norm decoder: RMS norm,
//! grouped-query attention with optional Q/K/V biases, `RoPE`, and a gated feed-forward. The managed caller owns the memory-mapped weights and the `RoPE` tables; this module validates every
//! shape and byte range before a backend reads them.

mod checks;
mod encoding;
mod segments;

pub use encoding::{
    ALL_Q8_0, ENCODING_Q4_K, ENCODING_Q6_K, ENCODING_Q8_0, Encoding, Matrix, unpack,
};

use super::error::GpuError;
use checks::{RangeCheck, check_layer, validate_shape};

/// Bytes in one `Q8_0` block: an FP16 scale and 32 signed codes.
pub const Q8_0_BLOCK_BYTES: u64 = 34;
/// Values in one `Q8_0` block.
pub const Q8_0_BLOCK_VALUES: u32 = 32;
/// Head dimensions the Metal kernels compile for (Qwen2.5-0.5B uses 64, Qwen2.5-7B and larger use 128).
pub const METAL_HEAD_DIMS: [u32; 2] = [64, 128];
/// Head dimension the CUDA kernels are written for.
pub const CUDA_HEAD_DIM: u32 = 64;
/// Largest number of KV slots a model instance can own.
pub const MAX_SLOTS: u32 = 65;
/// Largest query-head group per KV head (one simdgroup row block).
pub const MAX_GROUP: u32 = 8;
/// Offset of a tensor the architecture does not have (for example a bias).
pub const NO_TENSOR: u64 = u64::MAX;
/// Rotate the first half of a head against the second half.
pub const ROPE_NEOX: u32 = 0;
/// `silu(gate) * up`.
pub const ACTIVATION_SWIGLU: u32 = 0;
/// FP32 KV cache.
pub const KV_F32: u32 = 0;
/// FP16 KV cache with FP32 accumulation (explicit profile).
pub const KV_F16: u32 = 1;

/// Element type of a KV slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    F32,
    F16,
}

impl KvPrecision {
    /// Bytes per stored key or value element.
    #[must_use]
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Byte offsets of one transformer block's tensors inside the mapped weight file; optional tensors use
/// [`NO_TENSOR`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecoderLayerOffsets {
    pub attention_norm: u64,
    pub query: u64,
    pub key: u64,
    pub value: u64,
    pub query_bias: u64,
    pub key_bias: u64,
    pub value_bias: u64,
    pub query_norm: u64,
    pub key_norm: u64,
    pub attention_output: u64,
    pub feed_forward_norm: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    /// GGML type IDs of Q, K, V, O, gate, up, and down, four bits each from the lowest (ADR-021).
    pub encodings: u64,
}

impl DecoderLayerOffsets {
    /// The encoding of one of this layer's matrices.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] for an unimplemented encoding.
    pub fn encoding(&self, matrix: Matrix) -> Result<Encoding, GpuError> {
        unpack(self.encodings, matrix)
    }
}

const _: () = assert!(std::mem::size_of::<DecoderLayerOffsets>() == 120);

/// The C ABI model description (ADR-012). Pointers stay owned by the caller for the model's lifetime.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DecoderDesc {
    pub weights: *const u8,
    pub weights_length: u64,
    pub layers: *const DecoderLayerOffsets,
    pub rope_cosines: *const f32,
    pub rope_sines: *const f32,
    pub token_embedding: u64,
    pub output_norm: u64,
    pub output: u64,
    pub layer_count: u32,
    pub hidden: u32,
    pub feed_forward: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocabulary: u32,
    pub context: u32,
    pub session_slots: u32,
    pub step_tokens: u32,
    pub logits_rows: u32,
    pub rms_epsilon: f32,
    /// GGML type ID of the token embedding (ADR-021).
    pub embedding_encoding: u32,
    pub rope_layout: u32,
    pub activation: u32,
    pub kv_precision: u32,
    /// KV slots grow in multiples of this many positions, at least doubling (ADR-017); a positive multiple of 64.
    pub kv_growth_positions: u32,
    /// GGML type ID of the output projection (ADR-021).
    pub output_encoding: u32,
}

// The C ABI descriptor layout is mirrored in C# (`NativeDecoderDesc`); both sides pin its size.
const _: () = assert!(std::mem::size_of::<DecoderDesc>() == 136);

/// One token of a batched step: its KV slot, token ID, position, and logits row or `-1` (ADR-007).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchToken {
    pub slot: i32,
    pub token: i32,
    pub position: i32,
    pub logits_row: i32,
}

/// Validated dimensions of a decoder instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderShape {
    pub layer_count: u32,
    pub hidden: u32,
    pub feed_forward: u32,
    pub heads: u32,
    pub kv_heads: u32,
    /// 64 or 128 on Metal (compiled per model), 64 on CUDA.
    pub head_dim: u32,
    pub vocabulary: u32,
    pub context: u32,
    pub session_slots: u32,
    pub step_tokens: u32,
    pub logits_rows: u32,
    pub rms_epsilon: f32,
    pub kv_precision: KvPrecision,
    pub kv_growth: u32,
}

impl DecoderShape {
    /// Query heads per KV head.
    #[must_use]
    pub const fn group(&self) -> u32 {
        self.heads / self.kv_heads
    }

    /// Width of the packed K or V projection.
    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Floats in one row of the attention partial buffer: the head's output values, the running maximum, and
    /// the running sum.
    #[must_use]
    pub const fn partial_row_floats(&self) -> u32 {
        self.head_dim + 2
    }

    /// Floats in one token row of the packed Q|K|V projection.
    #[must_use]
    pub const fn qkv_width(&self) -> u32 {
        self.hidden + 2 * self.kv_width()
    }

    /// Bytes of K and V one position occupies across layers and KV heads.
    #[must_use]
    pub const fn kv_bytes_per_position(&self) -> u64 {
        2 * self.layer_count as u64
            * self.kv_heads as u64
            * self.head_dim as u64
            * self.kv_precision.element_bytes()
    }

    /// Bytes of one slot's K and V cache for `capacity` positions.
    #[must_use]
    pub const fn slot_bytes(&self, capacity: u32) -> u64 {
        self.kv_bytes_per_position() * capacity as u64
    }

    /// ADR-017 reservation: exactly the positions a request will use, rounded up to the growth unit and capped at
    /// the context, so a known prompt and output length allocate once.
    #[must_use]
    pub const fn reserve_capacity(&self, positions: u32) -> u32 {
        let rounded = positions
            .div_ceil(self.kv_growth)
            .saturating_mul(self.kv_growth);
        if rounded < self.context {
            rounded
        } else {
            self.context
        }
    }

    /// ADR-017 growth: the need rounded up to the growth unit, at least double the current capacity, never
    /// beyond the context.
    #[must_use]
    pub fn next_capacity(&self, needed: u32, current: u32) -> u32 {
        let rounded = needed
            .div_ceil(self.kv_growth)
            .saturating_mul(self.kv_growth);
        rounded.max(current.saturating_mul(2)).min(self.context)
    }
}

/// A validated description: shape, per-layer offsets, and the `RoPE` table length in floats.
#[derive(Debug, Clone)]
pub struct DecoderPlan {
    pub shape: DecoderShape,
    pub layers: Vec<DecoderLayerOffsets>,
    pub token_embedding: u64,
    pub output_norm: u64,
    pub output: u64,
    pub weights_length: u64,
    pub rope_floats: usize,
    pub embedding_encoding: Encoding,
    pub output_encoding: Encoding,
}

impl DecoderPlan {
    /// Validates dimensions and every tensor byte range against the mapped weight length.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidArgument`] for inconsistent shapes, out-of-range or misaligned offsets, and
    /// [`GpuError::Unavailable`] for shapes the kernels do not implement.
    pub fn new(desc: &DecoderDesc, layers: &[DecoderLayerOffsets]) -> Result<Self, GpuError> {
        let shape = validate_shape(desc)?;
        if layers.len() != shape.layer_count as usize {
            return Err(GpuError::invalid(
                "layer offset count differs from the layer count",
            ));
        }

        let ranges = RangeCheck {
            length: desc.weights_length,
        };
        let (hidden, vocab) = (u64::from(shape.hidden), u64::from(shape.vocabulary));
        let embedding_encoding = Encoding::from_id(desc.embedding_encoding)?;
        let output_encoding = Encoding::from_id(desc.output_encoding)?;
        ranges.matrix(
            desc.token_embedding,
            vocab,
            hidden,
            embedding_encoding,
            "token_embd",
        )?;
        ranges.matrix(desc.output, vocab, hidden, output_encoding, "output")?;
        ranges.f32(desc.output_norm, hidden, "output_norm")?;
        for (index, layer) in layers.iter().enumerate() {
            check_layer(&ranges, &shape, layer).map_err(|error| match error {
                GpuError::InvalidArgument(message) => {
                    GpuError::invalid(format!("layer {index}: {message}"))
                }
                other => other,
            })?;
        }

        let rope_floats = usize::try_from(u64::from(shape.context) * u64::from(shape.head_dim / 2))
            .map_err(|_| GpuError::invalid("RoPE table length overflows usize"))?;
        Ok(Self {
            shape,
            layers: layers.to_vec(),
            token_embedding: desc.token_embedding,
            output_norm: desc.output_norm,
            output: desc.output,
            weights_length: desc.weights_length,
            rope_floats,
            embedding_encoding,
            output_encoding,
        })
    }

    /// Validates one step and returns how many logits rows it fills.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidArgument`] for an empty or oversized step, an out-of-range slot, token, or
    /// position, or logits rows that are not exactly `0..k-1` in token order.
    pub fn validate_step(&self, tokens: &[BatchToken]) -> Result<u32, GpuError> {
        let shape = &self.shape;
        if tokens.is_empty() || tokens.len() > shape.step_tokens as usize {
            return Err(GpuError::invalid(format!(
                "a step holds 1..{} tokens",
                shape.step_tokens
            )));
        }

        let mut rows = 0_u32;
        for token in tokens {
            let in_range =
                |value: i32, limit: u32| u32::try_from(value).is_ok_and(|value| value < limit);
            if !in_range(token.slot, shape.session_slots)
                || !in_range(token.token, shape.vocabulary)
                || !in_range(token.position, shape.context)
            {
                return Err(GpuError::invalid(format!("invalid batch token {token:?}")));
            }

            if token.logits_row >= 0 {
                if u32::try_from(token.logits_row) != Ok(rows) || rows >= shape.logits_rows {
                    return Err(GpuError::invalid(
                        "logits rows must be 0..k-1 in token order",
                    ));
                }

                rows += 1;
            }
        }

        Ok(rows)
    }
}

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod tests;
