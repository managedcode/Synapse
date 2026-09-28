use super::q8_0::{BLOCK_BYTES, BLOCK_ELEMENTS, Q8Activations, Q8Weights, fp16_to_f32};

/// Portable kernel over validated slices.
pub fn matmul(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    stride: usize,
) {
    let blocks = weights.blocks_per_row;
    let row_bytes = blocks * BLOCK_BYTES;
    let row_codes = blocks * BLOCK_ELEMENTS;
    for (row, weights_row) in weights
        .data
        .chunks_exact(row_bytes)
        .take(weights.rows)
        .enumerate()
    {
        for token in 0..activations.tokens {
            output[token * stride + row] = dot_row(
                weights_row,
                &activations.quants[token * row_codes..(token + 1) * row_codes],
                &activations.scales[token * blocks..(token + 1) * blocks],
            );
        }
    }
}

fn dot_row(weights: &[u8], codes: &[i8], scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let blocks = weights
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(codes.as_chunks::<BLOCK_ELEMENTS>().0)
        .zip(scales);
    for ((block, values), &activation_scale) in blocks {
        let weight_scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dot: i32 = block[2..]
            .iter()
            .zip(values)
            .map(|(&weight, &value)| i32::from(i8::from_le_bytes([weight])) * i32::from(value))
            .sum();
        sum = exact_f32(dot).mul_add(weight_scale * activation_scale, sum);
    }

    sum
}

/// Converts a block dot product, whose magnitude is at most 32 * 128 * 128 < 2^24, exactly.
#[allow(clippy::cast_precision_loss)]
const fn exact_f32(value: i32) -> f32 {
    value as f32
}
