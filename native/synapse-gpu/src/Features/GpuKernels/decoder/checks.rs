//! Shape and byte-range validation of a decoder description.

use super::{
    ACTIVATION_SWIGLU, DecoderDesc, DecoderLayerOffsets, DecoderShape, Encoding, KV_F16, KV_F32,
    KvPrecision, MAX_GROUP, MAX_SLOTS, METAL_HEAD_DIMS, Matrix, NO_TENSOR, Q8_0_BLOCK_VALUES,
    ROPE_NEOX,
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
        head_dim: desc.head_dim,
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
        kv_growth: desc.kv_growth_positions,
    })
}

/// Encodings, `RoPE` layouts, and activations the kernels do not implement are `Unavailable`.
fn check_variants(desc: &DecoderDesc) -> Result<(), GpuError> {
    Encoding::from_id(desc.embedding_encoding)?;
    Encoding::from_id(desc.output_encoding)?;
    if desc.rope_layout != ROPE_NEOX || desc.activation != ACTIVATION_SWIGLU {
        return Err(GpuError::Unavailable(format!(
            "the GPU kernels implement NeoX RoPE and SwiGLU; the model asks for RoPE layout {}, activation {}",
            desc.rope_layout, desc.activation
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

    if desc.kv_growth_positions == 0 || !desc.kv_growth_positions.is_multiple_of(64) {
        return Err(GpuError::invalid(
            "the KV growth unit must be a positive multiple of 64 positions",
        ));
    }

    if !METAL_HEAD_DIMS.contains(&desc.head_dim) || desc.hidden != desc.heads * desc.head_dim {
        return Err(GpuError::Unavailable(format!(
            "the GPU kernels implement head dimensions {METAL_HEAD_DIMS:?} with hidden = heads x head; the model \
             uses head {} and hidden {}",
            desc.head_dim, desc.hidden
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
    ranges.matrix(
        layer.query,
        hidden,
        hidden,
        layer.encoding(Matrix::Query)?,
        "attn_q",
    )?;
    ranges.matrix(
        layer.key,
        kv,
        hidden,
        layer.encoding(Matrix::Key)?,
        "attn_k",
    )?;
    ranges.matrix(
        layer.value,
        kv,
        hidden,
        layer.encoding(Matrix::Value)?,
        "attn_v",
    )?;
    ranges.optional_f32(layer.query_bias, hidden, "attn_q.bias")?;
    ranges.optional_f32(layer.key_bias, kv, "attn_k.bias")?;
    ranges.optional_f32(layer.value_bias, kv, "attn_v.bias")?;
    if layer.query_norm != NO_TENSOR || layer.key_norm != NO_TENSOR {
        return Err(GpuError::Unavailable(
            "per-head query/key normalization is not implemented by the GPU kernels yet".into(),
        ));
    }

    ranges.matrix(
        layer.attention_output,
        hidden,
        hidden,
        layer.encoding(Matrix::Output)?,
        "attn_output",
    )?;
    ranges.f32(layer.feed_forward_norm, hidden, "ffn_norm")?;
    ranges.matrix(
        layer.gate,
        ffn,
        hidden,
        layer.encoding(Matrix::Gate)?,
        "ffn_gate",
    )?;
    ranges.matrix(layer.up, ffn, hidden, layer.encoding(Matrix::Up)?, "ffn_up")?;
    ranges.matrix(
        layer.down,
        hidden,
        ffn,
        layer.encoding(Matrix::Down)?,
        "ffn_down",
    )
}

pub(super) struct RangeCheck {
    pub length: u64,
}

impl RangeCheck {
    /// A `rows` x `columns` matrix in `encoding`; K-quant rows must be a whole number of 256-value blocks.
    pub fn matrix(
        &self,
        offset: u64,
        rows: u64,
        columns: u64,
        encoding: Encoding,
        name: &str,
    ) -> Result<(), GpuError> {
        if !columns.is_multiple_of(encoding.block_values()) {
            return Err(GpuError::Unavailable(format!(
                "tensor {name} has {columns} columns, not a multiple of its {}-value blocks",
                encoding.block_values()
            )));
        }

        let bytes = rows
            .checked_mul(columns / encoding.block_values())
            .and_then(|blocks| blocks.checked_mul(encoding.block_bytes()));
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
