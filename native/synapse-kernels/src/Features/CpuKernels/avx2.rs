use std::arch::x86_64::{
    __m256, __m256i, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    _mm256_castps256_ps128, _mm256_cvtepi32_ps, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_sign_epi8,
};

use super::q8_0::{
    BLOCK_BYTES, BLOCK_ELEMENTS, LONG_ROW_BLOCKS, Q8Activations, Q8Weights, fp16_to_f32,
};

/// AVX2/FMA kernel with the same row strategy as the managed x64 kernel.
///
/// # Safety
///
/// The CPU must support AVX2 and FMA, and the buffers must satisfy the shape checked by
/// `q8_0::validate`.
#[target_feature(enable = "avx2,fma")]
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
                for (lane, value) in sums.into_iter().enumerate() {
                    *out.add(token * stride + row + lane) = value;
                }
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

#[target_feature(enable = "avx2,fma")]
unsafe fn rows4(
    row0: *const u8,
    row_bytes: usize,
    blocks: usize,
    codes: *const i8,
    scales: *const f32,
) -> [f32; 4] {
    let mut sums = [_mm256_setzero_ps(); 4];
    // SAFETY: the caller guarantees four rows of `blocks` blocks and matching activations.
    unsafe {
        for block in 0..blocks {
            let values = load(codes.add(block * BLOCK_ELEMENTS).cast::<u8>());
            let activation_scale = *scales.add(block);
            for (lane, sum) in sums.iter_mut().enumerate() {
                let weights = row0.add(lane * row_bytes + block * BLOCK_BYTES);
                *sum = accumulate(*sum, weights, values, activation_scale);
            }
        }
    }

    sums.map(|sum| horizontal_sum(sum))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn row1(row: *const u8, blocks: usize, codes: *const i8, scales: *const f32) -> f32 {
    let mut sums = [_mm256_setzero_ps(); 2];
    let unrolled = blocks & !1;
    // SAFETY: the caller guarantees one row of `blocks` blocks and matching activations.
    unsafe {
        for block in (0..unrolled).step_by(2) {
            for (lane, sum) in sums.iter_mut().enumerate() {
                let index = block + lane;
                let values = load(codes.add(index * BLOCK_ELEMENTS).cast::<u8>());
                *sum = accumulate(
                    *sum,
                    row.add(index * BLOCK_BYTES),
                    values,
                    *scales.add(index),
                );
            }
        }

        for block in unrolled..blocks {
            let values = load(codes.add(block * BLOCK_ELEMENTS).cast::<u8>());
            sums[0] = accumulate(
                sums[0],
                row.add(block * BLOCK_BYTES),
                values,
                *scales.add(block),
            );
        }
    }

    horizontal_sum(sums[0]) + horizontal_sum(sums[1])
}

#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate(
    sum: __m256,
    block: *const u8,
    values: __m256i,
    activation_scale: f32,
) -> __m256 {
    // SAFETY: the caller guarantees `block` points at a complete 34-byte Q8_0 block.
    unsafe {
        let weights = load(block.add(2));
        let magnitudes = _mm256_sign_epi8(weights, weights);
        let signed_values = _mm256_sign_epi8(values, weights);
        let pairs = _mm256_maddubs_epi16(magnitudes, signed_values);
        let dots = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        let weight_scale = fp16_to_f32(u16::from_le_bytes([*block, *block.add(1)]));
        _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(dots),
            _mm256_set1_ps(weight_scale * activation_scale),
            sum,
        )
    }
}

#[target_feature(enable = "avx2,fma")]
fn horizontal_sum(value: __m256) -> f32 {
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let quad = _mm_add_ps(low, high);
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)))
}

/// Unaligned 32-byte load; `_mm256_loadu_si256` has no alignment requirement.
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)]
unsafe fn load(source: *const u8) -> __m256i {
    // SAFETY: the caller guarantees 32 readable bytes at `source`; the load is unaligned.
    unsafe { _mm256_loadu_si256(source.cast::<__m256i>()) }
}
