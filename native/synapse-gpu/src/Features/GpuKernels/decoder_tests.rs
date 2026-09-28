use super::{
    BatchToken, DecoderDesc, DecoderLayerOffsets, DecoderPlan, ENCODING_Q8_0, KV_F32, NO_TENSOR,
    Q8_0_BLOCK_BYTES,
};
use crate::gpu_kernels::error::GpuError;

const HIDDEN: u64 = 128;
const KV: u64 = 64;
const FFN: u64 = 256;
const VOCAB: u64 = 16;

fn q8_bytes(rows: u64, columns: u64) -> u64 {
    rows * columns / 32 * Q8_0_BLOCK_BYTES
}

/// Lays tensors out back to back (each padded to 32 bytes) and returns the offsets plus the total length.
fn layout() -> (DecoderLayerOffsets, u64, u64, u64, u64) {
    let mut cursor = 0_u64;
    let mut take = |bytes: u64| {
        let offset = cursor;
        cursor += bytes.div_ceil(32) * 32;
        offset
    };
    let embedding = take(q8_bytes(VOCAB, HIDDEN));
    let output = take(q8_bytes(VOCAB, HIDDEN));
    let norm = take(HIDDEN * 4);
    let layer = DecoderLayerOffsets {
        attention_norm: take(HIDDEN * 4),
        query: take(q8_bytes(HIDDEN, HIDDEN)),
        key: take(q8_bytes(KV, HIDDEN)),
        value: take(q8_bytes(KV, HIDDEN)),
        query_bias: take(HIDDEN * 4),
        key_bias: take(KV * 4),
        value_bias: take(KV * 4),
        query_norm: NO_TENSOR,
        key_norm: NO_TENSOR,
        attention_output: take(q8_bytes(HIDDEN, HIDDEN)),
        feed_forward_norm: take(HIDDEN * 4),
        gate: take(q8_bytes(FFN, HIDDEN)),
        up: take(q8_bytes(FFN, HIDDEN)),
        down: take(q8_bytes(HIDDEN, FFN)),
    };
    (layer, embedding, output, norm, cursor)
}

fn desc(length: u64, embedding: u64, output: u64, norm: u64) -> DecoderDesc {
    DecoderDesc {
        weights: std::ptr::null(),
        weights_length: length,
        layers: std::ptr::null(),
        rope_cosines: std::ptr::null(),
        rope_sines: std::ptr::null(),
        token_embedding: embedding,
        output_norm: norm,
        output,
        layer_count: 1,
        hidden: 128,
        feed_forward: 256,
        heads: 2,
        kv_heads: 1,
        head_dim: 64,
        vocabulary: 16,
        context: 8,
        session_slots: 3,
        step_tokens: 4,
        logits_rows: 2,
        rms_epsilon: 1e-6,
        matrix_encoding: ENCODING_Q8_0,
        rope_layout: 0,
        activation: 0,
        kv_precision: KV_F32,
    }
}

#[test]
fn plan_accepts_exact_layout() {
    let (layer, embedding, output, norm, length) = layout();
    let plan =
        DecoderPlan::new(&desc(length, embedding, output, norm), &[layer]).expect("valid layout");

    assert_eq!(plan.shape.group(), 2);
    assert_eq!(plan.shape.qkv_width(), 256);
    assert_eq!(plan.rope_floats, 8 * 32);
    assert_eq!(plan.shape.slot_bytes(), 2 * 8 * 64 * 4);
}

#[test]
fn plan_rejects_out_of_range_and_misaligned_tensors() {
    let (layer, embedding, output, norm, length) = layout();
    let short = DecoderPlan::new(&desc(length - 1, embedding, output, norm), &[layer]);
    let misaligned = DecoderPlan::new(
        &desc(length, embedding, output, norm),
        &[DecoderLayerOffsets {
            query_bias: layer.query_bias + 2,
            ..layer
        }],
    );
    let missing_layer = DecoderPlan::new(&desc(length, embedding, output, norm), &[]);

    assert!(matches!(short, Err(GpuError::InvalidArgument(_))));
    assert!(
        matches!(misaligned, Err(GpuError::InvalidArgument(message)) if message.contains("attn_q.bias"))
    );
    assert!(matches!(missing_layer, Err(GpuError::InvalidArgument(_))));
}

#[test]
fn plan_reports_unsupported_head_shapes_as_unavailable() {
    let (layer, embedding, output, norm, length) = layout();
    let mut wide = desc(length, embedding, output, norm);
    wide.head_dim = 128;
    let mut big_group = desc(length, embedding, output, norm);
    big_group.heads = 18;
    big_group.hidden = 18 * 64;

    assert!(matches!(
        DecoderPlan::new(&wide, &[layer]),
        Err(GpuError::Unavailable(_))
    ));
    assert!(matches!(
        DecoderPlan::new(&big_group, &[layer]),
        Err(GpuError::Unavailable(_))
    ));
}

#[test]
fn step_validation_enforces_bounds_and_logits_order() {
    let (layer, embedding, output, norm, length) = layout();
    let plan =
        DecoderPlan::new(&desc(length, embedding, output, norm), &[layer]).expect("valid layout");
    let token = |slot, id, position, row| BatchToken {
        slot,
        token: id,
        position,
        logits_row: row,
    };

    assert_eq!(
        plan.validate_step(&[token(0, 1, 0, -1), token(0, 2, 1, 0)]),
        Ok(1)
    );
    assert_eq!(
        plan.validate_step(&[token(1, 1, 7, 0), token(2, 2, 3, 1)]),
        Ok(2)
    );
    for bad in [
        vec![],
        vec![token(3, 1, 0, 0)],
        vec![token(0, 16, 0, 0)],
        vec![token(0, 1, 8, 0)],
        vec![token(0, 1, 0, 1)],
        vec![token(0, 1, 0, 0), token(1, 1, 0, 0)],
        vec![token(0, 1, 0, -1); 5],
    ] {
        assert!(plan.validate_step(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn plan_accepts_architectures_without_attention_biases() {
    let (layer, embedding, output, norm, length) = layout();
    let unbiased = DecoderLayerOffsets {
        query_bias: NO_TENSOR,
        key_bias: NO_TENSOR,
        value_bias: NO_TENSOR,
        ..layer
    };

    let plan = DecoderPlan::new(&desc(length, embedding, output, norm), &[unbiased]);

    assert!(plan.is_ok());
}

#[test]
fn plan_reports_unimplemented_variants_as_unavailable() {
    let (layer, embedding, output, norm, length) = layout();
    let base = desc(length, embedding, output, norm);
    let q4 = DecoderDesc {
        matrix_encoding: 2,
        ..base
    };
    let interleaved = DecoderDesc {
        rope_layout: 1,
        ..base
    };
    let gelu = DecoderDesc {
        activation: 1,
        ..base
    };
    let qk_norm = DecoderLayerOffsets {
        query_norm: layer.attention_norm,
        ..layer
    };

    for (desc, layers) in [
        (q4, layer),
        (interleaved, layer),
        (gelu, layer),
        (base, qk_norm),
    ] {
        assert!(matches!(
            DecoderPlan::new(&desc, &[layers]),
            Err(GpuError::Unavailable(_))
        ));
    }
}

#[test]
fn fp16_kv_halves_slot_bytes_and_unknown_precision_is_rejected() {
    let (layer, embedding, output, norm, length) = layout();
    let base = desc(length, embedding, output, norm);
    let half = DecoderDesc {
        kv_precision: super::KV_F16,
        ..base
    };
    let unknown = DecoderDesc {
        kv_precision: 2,
        ..base
    };

    let full = DecoderPlan::new(&base, &[layer]).expect("valid layout");
    let halved = DecoderPlan::new(&half, &[layer]).expect("valid layout");

    assert_eq!(halved.shape.slot_bytes() * 2, full.shape.slot_bytes());
    assert!(matches!(
        DecoderPlan::new(&unknown, &[layer]),
        Err(GpuError::InvalidArgument(_))
    ));
}
