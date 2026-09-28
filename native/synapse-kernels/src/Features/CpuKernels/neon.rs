use std::arch::aarch64::{
    float32x4_t, int8x16_t, int32x4_t, vaddq_f32, vaddvq_f32, vandq_u32, vcvtq_f32_s32, vdotq_s32,
    vdupq_n_f32, vdupq_n_s32, vdupq_n_u32, vfmaq_f32, vld1q_s8, vmulq_n_f32, vorrq_u32, vpaddq_s32,
    vreinterpretq_f32_u32, vreinterpretq_u32_f32, vsetq_lane_u32, vshlq_n_u32, vst1q_f32,
};

use super::q8_0::{
    BLOCK_BYTES, BLOCK_ELEMENTS, LONG_ROW_BLOCKS, Q8Activations, Q8Weights, fp16_to_f32,
};

/// NEON `sdot` kernel with the same row strategy as the managed ARM64 kernel.
///
/// # Safety
///
/// The CPU must support NEON and dotprod, and the buffers must satisfy the shape checked by
/// `q8_0::validate`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul(
    weights: &Q8Weights<'_>,
    activations: &Q8Activations<'_>,
    output: &mut [f32],
    stride: usize,
) {
    let blocks = weights.blocks_per_row;
    let row_bytes = blocks * BLOCK_BYTES;
    let row_codes = blocks * BLOCK_ELEMENTS;
    let quad_rows = if blocks >= LONG_ROW_BLOCKS {
        0
    } else {
        weights.rows & !3
    };
    let data = weights.data.as_ptr();
    let codes = activations.quants.as_ptr();
    let scales = activations.scales.as_ptr();
    let out = output.as_mut_ptr();
    for row in (0..quad_rows).step_by(4) {
        for token in 0..activations.tokens {
            // SAFETY: rows `row..row + 4` and token `token` lie inside the validated buffers.
            unsafe {
                let sums = rows4(
                    data.add(row * row_bytes),
                    row_bytes,
                    blocks,
                    codes.add(token * row_codes),
                    scales.add(token * blocks),
                );
                vst1q_f32(out.add(token * stride + row), sums);
            }
        }
    }

    for row in quad_rows..weights.rows {
        for token in 0..activations.tokens {
            // SAFETY: row `row` and token `token` lie inside the validated buffers.
            unsafe {
                *out.add(token * stride + row) = row1(
                    data.add(row * row_bytes),
                    blocks,
                    codes.add(token * row_codes),
                    scales.add(token * blocks),
                );
            }
        }
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn rows4(
    row0: *const u8,
    row_bytes: usize,
    blocks: usize,
    codes: *const i8,
    scales: *const f32,
) -> float32x4_t {
    let mut sum = vdupq_n_f32(0.0);
    // SAFETY: the caller guarantees four rows of `blocks` blocks and matching activations.
    unsafe {
        let row1 = row0.add(row_bytes);
        let row2 = row1.add(row_bytes);
        let row3 = row2.add(row_bytes);
        for block in 0..blocks {
            let offset = block * BLOCK_BYTES;
            let low = vld1q_s8(codes.add(block * BLOCK_ELEMENTS));
            let high = vld1q_s8(codes.add(block * BLOCK_ELEMENTS + 16));
            let dots = vpaddq_s32(
                vpaddq_s32(
                    dot(row0.add(offset), low, high),
                    dot(row1.add(offset), low, high),
                ),
                vpaddq_s32(
                    dot(row2.add(offset), low, high),
                    dot(row3.add(offset), low, high),
                ),
            );
            let weight_scales = scales4([
                header(row0.add(offset)),
                header(row1.add(offset)),
                header(row2.add(offset)),
                header(row3.add(offset)),
            ]);
            let combined = vmulq_n_f32(weight_scales, *scales.add(block));
            sum = vfmaq_f32(sum, vcvtq_f32_s32(dots), combined);
        }
    }

    sum
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn row1(row: *const u8, blocks: usize, codes: *const i8, scales: *const f32) -> f32 {
    let mut sums = [vdupq_n_f32(0.0); 4];
    let unrolled = blocks & !3;
    // SAFETY: the caller guarantees one row of `blocks` blocks and matching activations.
    unsafe {
        for block in (0..unrolled).step_by(4) {
            for (lane, sum) in sums.iter_mut().enumerate() {
                *sum = accumulate(*sum, row, codes, scales, block + lane);
            }
        }

        for block in unrolled..blocks {
            sums[0] = accumulate(sums[0], row, codes, scales, block);
        }
    }

    vaddvq_f32(vaddq_f32(
        vaddq_f32(sums[0], sums[1]),
        vaddq_f32(sums[2], sums[3]),
    ))
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn accumulate(
    sum: float32x4_t,
    row: *const u8,
    codes: *const i8,
    scales: *const f32,
    block: usize,
) -> float32x4_t {
    // SAFETY: the caller guarantees `block` is inside the row and its activation.
    unsafe {
        let weights = row.add(block * BLOCK_BYTES);
        let values = codes.add(block * BLOCK_ELEMENTS);
        let product = dot(weights, vld1q_s8(values), vld1q_s8(values.add(16)));
        let combined = scale(weights) * *scales.add(block);
        vfmaq_f32(sum, vcvtq_f32_s32(product), vdupq_n_f32(combined))
    }
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn dot(block: *const u8, low: int8x16_t, high: int8x16_t) -> int32x4_t {
    // SAFETY: the caller guarantees `block` points at a complete 34-byte Q8_0 block.
    unsafe {
        let codes = block.add(2).cast::<i8>();
        vdotq_s32(
            vdotq_s32(vdupq_n_s32(0), vld1q_s8(codes), low),
            vld1q_s8(codes.add(16)),
            high,
        )
    }
}

const unsafe fn header(block: *const u8) -> u16 {
    // SAFETY: the caller guarantees `block` points at a complete Q8_0 block header.
    unsafe { u16::from_le_bytes([*block, *block.add(1)]) }
}

unsafe fn scale(block: *const u8) -> f32 {
    // SAFETY: the caller guarantees `block` points at a complete Q8_0 block header.
    fp16_to_f32(unsafe { header(block) })
}

/// Vector form of [`fp16_to_f32`] for four finite scales, built with lane inserts.
#[target_feature(enable = "neon")]
fn scales4(bits: [u16; 4]) -> float32x4_t {
    let lanes = vsetq_lane_u32::<3>(
        u32::from(bits[3]),
        vsetq_lane_u32::<2>(
            u32::from(bits[2]),
            vsetq_lane_u32::<1>(u32::from(bits[1]), vdupq_n_u32(u32::from(bits[0]))),
        ),
    );
    let magnitude = vreinterpretq_f32_u32(vshlq_n_u32::<13>(vandq_u32(lanes, vdupq_n_u32(0x7fff))));
    let value = vmulq_n_f32(magnitude, f32::from_bits(0x7780_0000));
    let sign = vshlq_n_u32::<16>(vandq_u32(lanes, vdupq_n_u32(0x8000)));
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(value), sign))
}
