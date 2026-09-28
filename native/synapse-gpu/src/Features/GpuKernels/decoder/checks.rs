//! Shape and byte-range validation of a decoder description.

use super::{
    ACTIVATION_SWIGLU, DecoderDesc, DecoderLayerOffsets, DecoderShape, ENCODING_Q8_0, HEAD_DIM,
    KV_F16, KV_F32, KvPrecision, MAX_GROUP, MAX_SLOTS, NO_TENSOR, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_VALUES, ROPE_NEOX,
};
use crate::gpu_kernels::error::GpuError;

pub(super) fn validate_shape(desc: &DecoderDesc) -> Result<DecoderShape, GpuError> {
    check_variants(desc)?;
    check_dimensions(desc)?;
    Ok(DecoderShape {
        layer_count: desc.layer_count,
        hidden: desc.hidden,
        feed_forward: desc.feed_forward,
        heads: desc.heads,
        kv_heads: desc.kv_heads,
        vocabulary: desc.vocabulary,
        context: desc.context,
        session_slots: desc.session_slots,
        step_tokens: desc.step_tokens,
        logits_rows: desc.logits_rows,
        rms_epsilon: desc.rms_epsilon,
        kv_precision: match desc.kv_precision {
            KV_F32 => KvPrecision::F32,
            KV_F16 => KvPrecision::F16,
            other => return Err(GpuError::invalid(format!("unknown KV precision {other}"))),
        },
    })
}

/// Encodings, `RoPE` layouts, and activations the kernels do not implement are `Unavailable`.
fn check_variants(desc: &DecoderDesc) -> Result<(), GpuError> {
    if desc.matrix_encoding != ENCODING_Q8_0
        || desc.rope_layout != ROPE_NEOX
        || desc.activation != ACTIVATION_SWIGLU
    {
        return Err(GpuError::Unavailable(format!(
            "the GPU kernels implement Q8_0 matrices, NeoX RoPE, and SwiGLU; the model asks for encoding {}, \
             RoPE layout {}, activation {}",
            desc.matrix_encoding, desc.rope_layout, desc.activation
        )));
    }

    Ok(())
}

fn check_dimensions(desc: &DecoderDesc) -> Result<(), GpuError> {
    let positive = [
        desc.layer_count,
        desc.hidden,
        desc.feed_forward,
        desc.heads,
        desc.kv_heads,
        desc.vocabulary,
        desc.context,
        desc.session_slots,
        desc.step_tokens,
        desc.logits_rows,
    ];
    if positive.contains(&0) || !desc.rms_epsilon.is_finite() || desc.rms_epsilon <= 0.0 {
        return Err(GpuError::invalid(
            "every dimension must be positive and epsilon finite",
        ));
    }

    if desc.head_dim != HEAD_DIM || desc.hidden != desc.heads * HEAD_DIM {
        return Err(GpuError::Unavailable(format!(
            "the GPU kernels implement head dimension {HEAD_DIM}; the model uses {}",
            desc.head_dim
        )));
    }

    if !desc.heads.is_multiple_of(desc.kv_heads) || desc.heads / desc.kv_heads > MAX_GROUP {
        return Err(GpuError::Unavailable(format!(
            "query heads per KV head must divide evenly and be at most {MAX_GROUP}"
        )));
    }

    if !desc.hidden.is_multiple_of(Q8_0_BLOCK_VALUES)
        || !desc.feed_forward.is_multiple_of(Q8_0_BLOCK_VALUES)
    {
        return Err(GpuError::invalid(
            "hidden and feed-forward widths must be multiples of 32",
        ));
    }

    if desc.session_slots > MAX_SLOTS {
        return Err(GpuError::invalid(format!(
            "at most {MAX_SLOTS} KV slots are supported"
        )));
    }

    Ok(())
}

pub(super) fn check_layer(
    ranges: &RangeCheck,
    shape: &DecoderShape,
    layer: &DecoderLayerOffsets,
) -> Result<(), GpuError> {
    let (hidden, kv, ffn) = (
        u64::from(shape.hidden),
        u64::from(shape.kv_width()),
        u64::from(shape.feed_forward),
    );
    ranges.f32(layer.attention_norm, hidden, "attn_norm")?;
    ranges.q8(layer.query, hidden, hidden, "attn_q")?;
    ranges.q8(layer.key, kv, hidden, "attn_k")?;
    ranges.q8(layer.value, kv, hidden, "attn_v")?;
    ranges.optional_f32(layer.query_bias, hidden, "attn_q.bias")?;
    ranges.optional_f32(layer.key_bias, kv, "attn_k.bias")?;
    ranges.optional_f32(layer.value_bias, kv, "attn_v.bias")?;
    if layer.query_norm != NO_TENSOR || layer.key_norm != NO_TENSOR {
        return Err(GpuError::Unavailable(
            "per-head query/key normalization is not implemented by the GPU kernels yet".into(),
        ));
    }

    ranges.q8(layer.attention_output, hidden, hidden, "attn_output")?;
    ranges.f32(layer.feed_forward_norm, hidden, "ffn_norm")?;
    ranges.q8(layer.gate, ffn, hidden, "ffn_gate")?;
    ranges.q8(layer.up, ffn, hidden, "ffn_up")?;
    ranges.q8(layer.down, hidden, ffn, "ffn_down")
}

pub(super) struct RangeCheck {
    pub length: u64,
}

impl RangeCheck {
    pub fn q8(&self, offset: u64, rows: u64, columns: u64, name: &str) -> Result<(), GpuError> {
        let bytes = rows
            .checked_mul(columns / u64::from(Q8_0_BLOCK_VALUES))
            .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES));
        self.span(offset, bytes, 2, name)
    }

    fn optional_f32(&self, offset: u64, values: u64, name: &str) -> Result<(), GpuError> {
        if offset == NO_TENSOR {
            Ok(())
        } else {
            self.f32(offset, values, name)
        }
    }

    pub fn f32(&self, offset: u64, values: u64, name: &str) -> Result<(), GpuError> {
        self.span(offset, values.checked_mul(4), 4, name)
    }

    fn span(
        &self,
        offset: u64,
        bytes: Option<u64>,
        alignment: u64,
        name: &str,
    ) -> Result<(), GpuError> {
        let end = bytes.and_then(|bytes| offset.checked_add(bytes));
        match end {
            Some(end) if end <= self.length && offset.is_multiple_of(alignment) => Ok(()),
            _ => Err(GpuError::invalid(format!(
                "tensor {name} at offset {offset} is outside or misaligned"
            ))),
        }
    }
}
