use super::{
    ALL_Q8_0, BatchToken, DecoderDesc, DecoderLayerOffsets, DecoderPlan, ENCODING_Q8_0, KV_F32,
    NO_TENSOR, Q8_0_BLOCK_BYTES,
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
        encodings: ALL_Q8_0,
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
        embedding_encoding: ENCODING_Q8_0,
        rope_layout: 0,
        activation: 0,
        kv_precision: KV_F32,
        kv_growth_positions: 64,
        output_encoding: ENCODING_Q8_0,
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
    assert_eq!(plan.shape.slot_bytes(8), 2 * 8 * 64 * 4);
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
        embedding_encoding: 2,
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

    assert_eq!(halved.shape.slot_bytes(8) * 2, full.shape.slot_bytes(8));
    assert!(matches!(
        DecoderPlan::new(&unknown, &[layer]),
        Err(GpuError::InvalidArgument(_))
    ));
}

#[test]
fn kv_capacity_grows_by_unit_and_doubling_up_to_the_context() {
    let (layer, embedding, output, norm, length) = layout();
    let mut description = desc(length, embedding, output, norm);
    description.context = 4_096;
    description.kv_growth_positions = 1_024;
    let shape = DecoderPlan::new(&description, &[layer])
        .expect("valid layout")
        .shape;

    assert_eq!(shape.next_capacity(1, 0), 1_024);
    assert_eq!(shape.next_capacity(1_025, 1_024), 2_048);
    assert_eq!(shape.next_capacity(3_001, 1_024), 3_072);
    assert_eq!(shape.next_capacity(3_500, 3_072), 4_096);
    assert_eq!(shape.kv_bytes_per_position(), 2 * 64 * 4);
}

#[test]
fn kv_growth_unit_must_be_a_positive_multiple_of_64() {
    let (layer, embedding, output, norm, length) = layout();
    for unit in [0, 32, 100] {
        let mut description = desc(length, embedding, output, norm);
        description.kv_growth_positions = unit;
        assert!(matches!(
            DecoderPlan::new(&description, &[layer]),
            Err(GpuError::InvalidArgument(_))
        ));
    }
}

/// Two layers with a 1 MiB hole between them, where a dropped layer's tensors would sit (ADR-019).
fn two_layers_with_gap(gap: u64) -> (DecoderPlan, u64) {
    let (layer, embedding, output, norm, length) = layout();
    let second_start = length + gap;
    let shift = |offset: u64| {
        if offset == NO_TENSOR {
            NO_TENSOR
        } else {
            offset - layer.attention_norm + second_start
        }
    };
    let second = DecoderLayerOffsets {
        attention_norm: shift(layer.attention_norm),
        query: shift(layer.query),
        key: shift(layer.key),
        value: shift(layer.value),
        query_bias: shift(layer.query_bias),
        key_bias: shift(layer.key_bias),
        value_bias: shift(layer.value_bias),
        query_norm: NO_TENSOR,
        key_norm: NO_TENSOR,
        attention_output: shift(layer.attention_output),
        feed_forward_norm: shift(layer.feed_forward_norm),
        gate: shift(layer.gate),
        up: shift(layer.up),
        down: shift(layer.down),
        encodings: ALL_Q8_0,
    };
    let total = second_start + (length - layer.attention_norm);
    let mut description = desc(total, embedding, output, norm);
    description.layer_count = 2;
    let plan = DecoderPlan::new(&description, &[layer, second]).expect("valid two-layer layout");
    (plan, length)
}

#[test]
fn weight_segments_leave_dropped_ranges_unmapped() {
    let page = 16_384;
    let (plan, first_end) = two_layers_with_gap(1 << 20);

    let segments = plan.weight_segments(page);

    assert_eq!(segments.len(), 2);
    assert!(
        segments
            .iter()
            .all(|&(start, end)| start % page == 0 && end % page == 0 && start < end)
    );
    assert!(segments[0].1 <= first_end.next_multiple_of(page));
    assert!(segments[1].0 >= first_end + (1 << 20) - page);
    assert_eq!(segments[1].1, plan.weights_length.next_multiple_of(page));
}

#[test]
fn weight_segments_merge_units_that_share_pages() {
    let (plan, _) = two_layers_with_gap(0);

    assert_eq!(
        plan.weight_segments(16_384),
        [(0, plan.weights_length.next_multiple_of(16_384))]
    );
}

/// A `Q4_K_M`-like layer (ADR-021): `Q4_K` except V and down in `Q6_K`, at hidden `hidden` and head size 128.
fn kquant_plan(length_delta: i64, hidden: u64) -> Result<DecoderPlan, GpuError> {
    use super::{ENCODING_Q4_K, ENCODING_Q6_K, Encoding};
    let (kv, ffn) = (128_u64, 512_u64);
    let (q4, q6) = (Encoding::Q4K, Encoding::Q6K);
    let mut cursor = 0_u64;
    let mut take = |bytes: u64| {
        let offset = cursor;
        cursor += bytes.div_ceil(32) * 32;
        offset
    };
    let embedding = take(q4.matrix_bytes(VOCAB, hidden));
    let output = take(q6.matrix_bytes(VOCAB, hidden));
    let norm = take(hidden * 4);
    let layer = DecoderLayerOffsets {
        attention_norm: take(hidden * 4),
        query: take(q4.matrix_bytes(hidden, hidden)),
        key: take(q4.matrix_bytes(kv, hidden)),
        value: take(q6.matrix_bytes(kv, hidden)),
        query_bias: take(hidden * 4),
        key_bias: take(kv * 4),
        value_bias: take(kv * 4),
        query_norm: NO_TENSOR,
        key_norm: NO_TENSOR,
        attention_output: take(q4.matrix_bytes(hidden, hidden)),
        feed_forward_norm: take(hidden * 4),
        gate: take(q4.matrix_bytes(ffn, hidden)),
        up: take(q4.matrix_bytes(ffn, hidden)),
        down: take(q6.matrix_bytes(hidden, ffn)),
        // Q, K, V, O, gate, up, down from the lowest four bits: C C E C C C E.
        encodings: 0x0ECC_CECC,
    };
    let mut description = desc(
        cursor.saturating_add_signed(length_delta),
        embedding,
        output,
        norm,
    );
    description.hidden = u32::try_from(hidden).unwrap_or(u32::MAX);
    description.feed_forward = 512;
    description.heads = 2;
    description.head_dim = 128;
    description.embedding_encoding = ENCODING_Q4_K;
    description.output_encoding = ENCODING_Q6_K;
    DecoderPlan::new(&description, &[layer])
}

#[test]
fn plan_sizes_kquant_matrices_by_their_blocks() {
    use super::{Encoding, Matrix};
    let plan = kquant_plan(0, 256).expect("an exact Q4_K_M-like layout is valid");

    assert_eq!(plan.embedding_encoding, Encoding::Q4K);
    assert_eq!(plan.output_encoding, Encoding::Q6K);
    assert_eq!(
        plan.layers[0].encoding(Matrix::Value).ok(),
        Some(Encoding::Q6K)
    );
    assert_eq!(
        plan.layers[0].encoding(Matrix::Gate).ok(),
        Some(Encoding::Q4K)
    );
    assert!(matches!(
        kquant_plan(-1, 256),
        Err(GpuError::InvalidArgument(_))
    ));
}

#[test]
fn kquant_rows_must_hold_whole_super_blocks() {
    assert!(matches!(kquant_plan(0, 128), Err(GpuError::Unavailable(_))));
}
