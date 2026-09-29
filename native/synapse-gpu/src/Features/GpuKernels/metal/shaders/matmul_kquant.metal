// GGML K-quant weights (ADR-021): Q4_K (144 bytes) and Q6_K (210 bytes) super-blocks of 256 values. Kernels walk a
// row as 32-value chunks, like Q8_0 blocks: eight chunks per super-block. Every value dequantizes as ggml's
// dequantize_row_q4_K / dequantize_row_q6_K does (d * sc * q - dmin * m, and d * sc * (q - 32)), in FP32.

constant constexpr ushort SYNAPSE_Q4K = 12;
constant constexpr ushort SYNAPSE_Q6K = 14;

static inline uint kq_bytes(ushort encoding) {
    return encoding == SYNAPSE_Q4K ? 144 : 210;
}

static inline float kq_half(device const uchar * p) {
    return (float)as_type<half>((ushort)(p[0] | (p[1] << 8)));
}

// ggml get_scale_min_k4: sub-block j's 6-bit scale and minimum from the 12 packed bytes.
static inline float2 kq_scale_min(device const uchar * packed, ushort j) {
    if (j < 4) {
        return float2(packed[j] & 63, packed[j + 4] & 63);
    }

    return float2((packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4), (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
}

// Byte `index` of the 12 packed scale bytes held as three little-endian words.
static inline uint kq_byte(uint w0, uint w1, uint w2, ushort index) {
    const uint word = index < 4 ? w0 : (index < 8 ? w1 : w2);
    return (word >> (8 * (index % 4))) & 0xFF;
}

// kq_scale_min over the 12 packed bytes held as three words.
static inline float2 kq_scale_min_words(uint w0, uint w1, uint w2, ushort j) {
    if (j < 4) {
        return float2(kq_byte(w0, w1, w2, j) & 63, kq_byte(w0, w1, w2, j + 4) & 63);
    }

    const uint scale = (kq_byte(w0, w1, w2, j + 4) & 0xF) | ((kq_byte(w0, w1, w2, j - 4) >> 6) << 4);
    const uint minimum = (kq_byte(w0, w1, w2, j + 4) >> 4) | ((kq_byte(w0, w1, w2, j) >> 6) << 4);
    return float2(scale, minimum);
}

// Values [lq * 8, lq * 8 + 8) of 32-value chunk `chunk` of a K-quant row.
template <ushort E>
static inline void kq_values8(device const uchar * row, uint chunk, ushort lq, thread float (&w)[8]) {
    device const uchar * block = row + (ulong)(chunk / 8) * kq_bytes(E);
    const ushort c = chunk % 8;
    if (E == SYNAPSE_Q4K) {
        const float2 sm = kq_scale_min(block + 4, c);
        const float d1 = kq_half(block) * sm.x;
        const float m1 = kq_half(block + 2) * sm.y;
        device const uchar * q = block + 16 + (c / 2) * 32 + lq * 8;
        for (short i = 0; i < 8; ++i) {
            w[i] = d1 * (float)((c % 2 == 0) ? (q[i] & 0xF) : (q[i] >> 4)) - m1;
        }
    } else {
        const ushort h = c / 4;
        const ushort quarter = c % 4;
        const float scale = kq_half(block + 208) * (float)as_type<char>(block[192 + h * 8 + quarter * 2 + lq / 2]);
        for (short i = 0; i < 8; ++i) {
            const ushort l = lq * 8 + i;
            const uchar low = block[h * 64 + ((quarter % 2 == 0) ? l : l + 32)];
            const int nibble = quarter < 2 ? (low & 0xF) : (low >> 4);
            const int high = (block[128 + h * 32 + l] >> (2 * quarter)) & 3;
            w[i] = scale * (float)((nibble | (high << 4)) - 32);
        }
    }
}

// A lane's eight codes of one chunk, decoded once and applied to every token: the lane's share of the chunk's dot
// product with x0|x1 is scale * dot(q, x) - offset * sum(x) (Q4_K: scale d1 and offset m1; Q6_K: scale d * sc and
// no offset). Codes load as one 8-byte vector, and a row's result depends only on the lane mapping, never on the
// token count.
struct kq_chunk {
    float4 q0;
    float4 q1;
    float scale;
    float offset;
};

template <ushort E>
static inline kq_chunk kq_decode8(device const uchar * row, uint chunk, ushort lq) {
    device const uchar * block = row + (ulong)(chunk / 8) * kq_bytes(E);
    const ushort c = chunk % 8;
    kq_chunk k;
    if (E == SYNAPSE_Q4K) {
        // Q4_K blocks are 144 bytes, a multiple of 16, so the header and codes load as aligned words.
        const half2 dm = *(device const half2 *)block;
        device const uint * words = (device const uint *)(block + 4);
        const float2 sm = kq_scale_min_words(words[0], words[1], words[2], c);
        const uint2 packed = *(device const uint2 *)(block + 16 + (c / 2) * 32 + lq * 8);
        const uint shift = (c % 2) * 4;
        k.q0 = float4((packed.x >> shift) & 0xF, (packed.x >> (8 + shift)) & 0xF,
                      (packed.x >> (16 + shift)) & 0xF, (packed.x >> (24 + shift)) & 0xF);
        k.q1 = float4((packed.y >> shift) & 0xF, (packed.y >> (8 + shift)) & 0xF,
                      (packed.y >> (16 + shift)) & 0xF, (packed.y >> (24 + shift)) & 0xF);
        k.scale = (float)dm.x * sm.x;
        k.offset = (float)dm.y * sm.y;
        return k;
    }

    const ushort h = c / 4;
    const ushort quarter = c % 4;
    const ushort l0 = lq * 8;
    device const uchar * low = block + h * 64 + ((quarter % 2 == 0) ? l0 : l0 + 32);
    device const uchar * high = block + 128 + h * 32 + l0;
    const uint4 low0 = uint4(uchar4(*(device const packed_uchar4 *)low));
    const uint4 low1 = uint4(uchar4(*(device const packed_uchar4 *)(low + 4)));
    const uint4 high0 = uint4(uchar4(*(device const packed_uchar4 *)high));
    const uint4 high1 = uint4(uchar4(*(device const packed_uchar4 *)(high + 4)));
    const uint low_shift = quarter < 2 ? 0 : 4;
    const uint high_shift = 2 * quarter;
    k.q0 = float4(int4(((low0 >> low_shift) & 0xF) | (((high0 >> high_shift) & 3) << 4)) - 32);
    k.q1 = float4(int4(((low1 >> low_shift) & 0xF) | (((high1 >> high_shift) & 3) << 4)) - 32);
    // Q6_K blocks are 210 bytes, so d at byte 208 is always 2-byte aligned.
    k.scale = (float)*(device const half *)(block + 208) * (float)as_type<char>(block[192 + h * 8 + quarter * 2 + lq / 2]);
    k.offset = 0.0f;
    return k;
}

static inline float kq_apply(thread const kq_chunk & k, float4 x0, float4 x1) {
    const float qx = dot(k.q0, x0) + dot(k.q1, x1);
    const float sx = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
    return k.scale * qx - k.offset * sx;
}

template <short R>
struct kq_rows {
    device const uchar * row[R];
    uint out_column[R];
    uint local_row[R];
    ulong bias[R];
    bool valid[R];
};

template <ushort E, short R>
static inline kq_rows<R> kq_locate(constant matmul_args & a, device const char * weights, uint first_row) {
    const ulong row_bytes = (ulong)(a.columns / 256) * kq_bytes(E);
    kq_rows<R> rows;
    for (short r = 0; r < R; ++r) {
        const uint row = first_row + r;
        rows.valid[r] = row < a.total_rows;
        uint local = rows.valid[r] ? row : 0;
        const uint s = locate_segment(a, local);
        rows.row[r] = (device const uchar *)(weights + a.segment[s].weight + (ulong)local * row_bytes);
        rows.out_column[r] = a.segment[s].out_column + local;
        rows.local_row[r] = local;
        rows.bias[r] = a.segment[s].bias;
    }

    return rows;
}

// Grid: (ceil(total_rows / (MV_SIMDGROUPS * R)), ceil(tokens / T)) threadgroups of MV_SIMDGROUPS simdgroups. Each
// simdgroup owns R rows outright: its lanes stride over the row's chunks, and one simd_sum per row and token
// finishes the row, with no threadgroup memory or barrier. A row's order depends only on the lane mapping, never
// on how many tokens share the dispatch.
template <ushort E, short T, short R>
kernel void synapse_kq_matvec(
        constant matmul_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint first_token = tg.y * T;
    const uint tokens = min((uint)T, a.tokens - first_token);
    const kq_rows<R> rows = kq_locate<E, R>(a, weights, (tg.x * MV_SIMDGROUPS + sg) * R);
    float sums[R][T];
    for (short r = 0; r < R; ++r) {
        for (short t = 0; t < T; ++t) {
            sums[r][t] = 0.0f;
        }
    }

    const short slot = lane / MV_LANES_PER_BLOCK;
    const short quarter = lane % MV_LANES_PER_BLOCK;
    device const float * x = input + (ulong)first_token * a.in_stride + quarter * 8;
    for (uint chunk = slot; chunk < a.columns / 32; chunk += 8) {
        float4 x0[T];
        float4 x1[T];
        for (short t = 0; t < T; ++t) {
            const uint source = min((uint)t, tokens - 1);
            device const float4 * v = (device const float4 *)(x + (ulong)source * a.in_stride + chunk * 32);
            x0[t] = v[0];
            x1[t] = v[1];
        }

        for (short r = 0; r < R; ++r) {
            const kq_chunk k = kq_decode8<E>(rows.row[r], rows.valid[r] ? chunk : 0, quarter);
            for (short t = 0; t < T; ++t) {
                sums[r][t] += kq_apply(k, x0[t], x1[t]);
            }
        }
    }

    for (short r = 0; r < R; ++r) {
        for (short t = 0; t < T; ++t) {
            const float total = simd_sum(sums[r][t]);
            if (lane != 0 || !rows.valid[r] || (uint)t >= tokens) {
                continue;
            }

            const float biased = rows.bias[r] != SYNAPSE_NO_BIAS
                ? total + ((device const float *)(weights + rows.bias[r]))[rows.local_row[r]]
                : total;
            device float * destination = output + (ulong)(first_token + t) * a.out_stride + rows.out_column[r];
            *destination = a.accumulate != 0 ? *destination + biased : biased;
        }
    }
}

#define SYNAPSE_KQ_MATVEC(NAME, E, T, R) \
    template [[host_name(NAME)]] kernel void synapse_kq_matvec<E, T, R>( \
        constant matmul_args &, device const char *, device const float *, device float *, uint2, ushort, ushort);

SYNAPSE_KQ_MATVEC("synapse_q4k_matvec_1", SYNAPSE_Q4K, 1, 1)
SYNAPSE_KQ_MATVEC("synapse_q4k_matvec_2", SYNAPSE_Q4K, 2, 1)
SYNAPSE_KQ_MATVEC("synapse_q4k_matvec_4", SYNAPSE_Q4K, 4, 1)
SYNAPSE_KQ_MATVEC("synapse_q4k_matvec_8", SYNAPSE_Q4K, 8, 1)
SYNAPSE_KQ_MATVEC("synapse_q6k_matvec_1", SYNAPSE_Q6K, 1, 1)
SYNAPSE_KQ_MATVEC("synapse_q6k_matvec_2", SYNAPSE_Q6K, 2, 1)
SYNAPSE_KQ_MATVEC("synapse_q6k_matvec_4", SYNAPSE_Q6K, 4, 1)
SYNAPSE_KQ_MATVEC("synapse_q6k_matvec_8", SYNAPSE_Q6K, 8, 1)

// The Q8_0 GEMM's tile walk with K-quant staging: each thread dequantizes 16 values of its weight row per chunk.
template <ushort E>
kernel void synapse_kq_gemm(
        constant matmul_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tile[GEMM_ROWS * GEMM_K + GEMM_TOKENS * GEMM_K];
    threadgroup float * wt = tile;
    threadgroup float * xt = tile + GEMM_ROWS * GEMM_K;
    const uint first_row = tg.x * GEMM_ROWS;
    const ushort load_row = tid / 2;
    const ushort load_half = tid % 2;
    const bool weight_valid = first_row + load_row < a.total_rows;
    uint local = weight_valid ? first_row + load_row : 0;
    const uint s = locate_segment(a, local);
    device const uchar * row =
        (device const uchar *)(weights + a.segment[s].weight + (ulong)local * ((a.columns / 256) * kq_bytes(E)));
    const ushort load_token = tid / 4;
    const ushort load_quarter = tid % 4;
    const bool token_valid = tg.y * GEMM_TOKENS + load_token < a.tokens;
    device const float * x =
        input + (ulong)(token_valid ? tg.y * GEMM_TOKENS + load_token : 0) * a.in_stride + load_quarter * 8;
    simdgroup_float8x8 acc[8];
    for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const ushort row_base = 32 * (sg % 2);
    const ushort token_base = 16 * (sg / 2);
    for (uint chunk = 0; chunk < a.columns / 32; ++chunk) {
        float w[8];
        threadgroup float * target = wt + load_row * GEMM_K + load_half * 16;
        for (short part = 0; part < 2; ++part) {
            kq_values8<E>(row, chunk, load_half * 2 + part, w);
            for (short i = 0; i < 8; ++i) {
                target[part * 8 + i] = weight_valid ? w[i] : 0.0f;
            }
        }

        device const float4 * v = (device const float4 *)(x + chunk * 32);
        threadgroup float4 * xtarget = (threadgroup float4 *)(xt + load_token * GEMM_K + load_quarter * 8);
        xtarget[0] = token_valid ? v[0] : float4(0.0f);
        xtarget[1] = token_valid ? v[1] : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_accumulate(acc, wt, xt, row_base, token_base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short j = 0; j < 2; ++j) {
        for (short i = 0; i < 4; ++i) {
            simdgroup_store(acc[j * 4 + i], tile + (token_base + j * 8) * GEMM_ROWS + row_base + i * 8, GEMM_ROWS);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_store(a, weights, output, tile, tg, tid);
}

template [[host_name("synapse_q4k_gemm")]] kernel void synapse_kq_gemm<SYNAPSE_Q4K>(
    constant matmul_args &, device const char *, device const float *, device float *, uint2, ushort, ushort);
template [[host_name("synapse_q6k_gemm")]] kernel void synapse_kq_gemm<SYNAPSE_Q6K>(
    constant matmul_args &, device const char *, device const float *, device float *, uint2, ushort, ushort);

// One thread per (column, token): the token's K-quant embedding row, dequantized.
template <ushort E>
kernel void synapse_kq_embed(
        constant embed_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const batch_token * tokens [[buffer(2)]],
        device float * hidden [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.columns || gid.y >= a.tokens) {
        return;
    }

    device const uchar * row = (device const uchar *)(weights + a.table + (ulong)tokens[gid.y].token * a.row_bytes);
    float w[8];
    kq_values8<E>(row, gid.x / 32, (gid.x % 32) / 8, w);
    hidden[(ulong)gid.y * a.out_stride + gid.x] = w[gid.x % 8];
}

template [[host_name("synapse_embed_q4k")]] kernel void synapse_kq_embed<SYNAPSE_Q4K>(
    constant embed_args &, device const char *, device const batch_token *, device float *, uint2);
template [[host_name("synapse_embed_q6k")]] kernel void synapse_kq_embed<SYNAPSE_Q6K>(
    constant embed_args &, device const char *, device const batch_token *, device float *, uint2);
