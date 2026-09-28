// Q8_0 matrix times FP32 token rows. Numeric contract (ADR-012): weights stay Q8_0 in the mapped model,
// activations stay FP32, and each row accumulates sum_blocks(d * sum(q * x)) in FP32. Every multiply-add is an
// explicit fma in a fixed order, so one token's result does not depend on how many tokens share the dispatch
// or which template instance ran it.

constant constexpr short MV_ROWS = 2;
constant constexpr short MV_SIMDGROUPS = 4;
constant constexpr short MV_LANES_PER_BLOCK = 4;

struct matvec_rows {
    device const block_q8_0 * row[MV_ROWS];
    uint out_column[MV_ROWS];
    uint local_row[MV_ROWS];
    ulong bias[MV_ROWS];
    bool valid[MV_ROWS];
};

static inline matvec_rows matvec_locate(constant matmul_args & a, device const char * weights, uint first_row) {
    const ulong row_bytes = (ulong)(a.columns / 32) * sizeof(block_q8_0);
    matvec_rows rows;
    for (short r = 0; r < MV_ROWS; ++r) {
        const uint row = first_row + r;
        rows.valid[r] = row < a.total_rows;
        uint local = rows.valid[r] ? row : 0;
        const uint s = locate_segment(a, local);

        rows.row[r] = (device const block_q8_0 *)(weights + a.segment[s].weight + (ulong)local * row_bytes);
        rows.out_column[r] = a.segment[s].out_column + local;
        rows.local_row[r] = local;
        rows.bias[r] = a.segment[s].bias;
    }

    return rows;
}

// Eight codes times eight activations, accumulated left to right with fma. The codes load as two packed
// (byte-aligned) vectors; the int8-to-float conversion is exact, so the result equals scalar loads.
static inline float dot8(device const int8_t * q, float4 x0, float4 x1) {
    const float4 a = float4(*(device const packed_char4 *)q);
    const float4 b = float4(*(device const packed_char4 *)(q + 4));
    float acc = a.x * x0.x;
    acc = fma(a.y, x0.y, acc);
    acc = fma(a.z, x0.z, acc);
    acc = fma(a.w, x0.w, acc);
    acc = fma(b.x, x1.x, acc);
    acc = fma(b.y, x1.y, acc);
    acc = fma(b.z, x1.z, acc);
    acc = fma(b.w, x1.w, acc);
    return acc;
}

// Reduces the lane sums across the simdgroups in a fixed order and writes each row with bias and residual.
template <short T>
static inline void matvec_finish(
        constant matmul_args & a, device const char * weights, device float * output, thread const matvec_rows & rows,
        thread const float (&sums)[MV_ROWS][T], threadgroup float (&partial)[MV_SIMDGROUPS][MV_ROWS][T],
        uint first_token, uint tokens, ushort sg, ushort lane) {
    for (short r = 0; r < MV_ROWS; ++r) {
        for (short t = 0; t < T; ++t) {
            const float total = simd_sum(sums[r][t]);
            if (lane == 0) {
                partial[sg][r][t] = total;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint id = sg * 32 + lane;
    const short r = id / T;
    const short t = id % T;
    if (id >= MV_ROWS * T || !rows.valid[r] || (uint)t >= tokens) {
        return;
    }

    float total = ((partial[0][r][t] + partial[1][r][t]) + partial[2][r][t]) + partial[3][r][t];
    if (rows.bias[r] != SYNAPSE_NO_BIAS) {
        total += ((device const float *)(weights + rows.bias[r]))[rows.local_row[r]];
    }

    device float * destination = output + (ulong)(first_token + t) * a.out_stride + rows.out_column[r];
    *destination = a.accumulate != 0 ? *destination + total : total;
}

// Grid: (ceil(total_rows / MV_ROWS), ceil(tokens / T)) threadgroups of MV_SIMDGROUPS * 32 threads. Output rows
// come from up to three weight segments (for example Q, K, and V) written to their own columns.
template <short T>
kernel void synapse_q8_matvec(
        constant matmul_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[MV_SIMDGROUPS][MV_ROWS][T];

    const uint first_token = tg.y * T;
    const uint tokens = min((uint)T, a.tokens - first_token);
    const uint blocks = a.columns / 32;
    const matvec_rows rows = matvec_locate(a, weights, tg.x * MV_ROWS);

    float sums[MV_ROWS][T];
    for (short r = 0; r < MV_ROWS; ++r) {
        for (short t = 0; t < T; ++t) {
            sums[r][t] = 0.0f;
        }
    }

    const short slot = lane / MV_LANES_PER_BLOCK;
    const short quarter = lane % MV_LANES_PER_BLOCK;
    device const float * x = input + (ulong)first_token * a.in_stride + quarter * 8;
    for (uint block = sg * 8 + slot; block < blocks; block += MV_SIMDGROUPS * 8) {
        float4 x0[T];
        float4 x1[T];
        for (short t = 0; t < T; ++t) {
            // Rows past `tokens` re-read the last valid row; their sums are never written.
            const uint source = min((uint)t, tokens - 1);
            device const float4 * v = (device const float4 *)(x + (ulong)source * a.in_stride + block * 32);
            x0[t] = v[0];
            x1[t] = v[1];
        }

        for (short r = 0; r < MV_ROWS; ++r) {
            device const block_q8_0 * b = rows.row[r] + (rows.valid[r] ? block : 0);
            const float d = (float)b->d;
            device const int8_t * q = b->qs + quarter * 8;
            for (short t = 0; t < T; ++t) {
                sums[r][t] = fma(dot8(q, x0[t], x1[t]), d, sums[r][t]);
            }
        }
    }

    matvec_finish<T>(a, weights, output, rows, sums, partial, first_token, tokens, sg, lane);
}

#define SYNAPSE_MATVEC(T) \
    template [[host_name("synapse_q8_matvec_" #T)]] kernel void synapse_q8_matvec<T>( \
        constant matmul_args &, device const char *, device const float *, device float *, \
        uint2, ushort, ushort);

SYNAPSE_MATVEC(1)
SYNAPSE_MATVEC(2)
SYNAPSE_MATVEC(4)
SYNAPSE_MATVEC(8)

// Prompt-run projections: C[token][row] = sum_k X[token][k] * W[row][k] with FP32 8x8 simdgroup matrices.
// A threadgroup of four simdgroups computes a tile of GEMM_ROWS weight rows by GEMM_TOKENS tokens; each K step
// is one Q8_0 block, dequantized exactly (d * q) into threadgroup memory. Each output element accumulates its
// K steps in the same order whatever tile or step it belongs to.
constant constexpr short GEMM_ROWS = 64;
constant constexpr short GEMM_TOKENS = 32;
constant constexpr short GEMM_K = 32;

// What one thread stages per K step: half a weight row's block (16 codes) and 8 activations of one token.
struct gemm_loader {
    device const block_q8_0 * weight_blocks;
    bool weight_valid;
    ushort load_row;
    ushort load_half;
    device const float * x;
    bool token_valid;
    ushort load_token;
    ushort load_quarter;
};

static inline gemm_loader gemm_prepare(constant matmul_args & a, device const char * weights,
                                       device const float * input, uint2 tg, ushort tid) {
    gemm_loader l;
    const uint first_row = tg.x * GEMM_ROWS;
    const uint first_token = tg.y * GEMM_TOKENS;
    const ulong row_bytes = (ulong)(a.columns / 32) * sizeof(block_q8_0);
    l.load_row = tid / 2;
    l.load_half = tid % 2;
    l.weight_valid = first_row + l.load_row < a.total_rows;
    uint local = l.weight_valid ? first_row + l.load_row : 0;
    const uint s = locate_segment(a, local);
    l.weight_blocks = (device const block_q8_0 *)(weights + a.segment[s].weight + (ulong)local * row_bytes);
    l.load_token = tid / 4;
    l.load_quarter = tid % 4;
    l.token_valid = first_token + l.load_token < a.tokens;
    l.x = input + (ulong)(l.token_valid ? first_token + l.load_token : 0) * a.in_stride + l.load_quarter * 8;
    return l;
}

// One K step of one thread held in registers: the next step is fetched while the current one multiplies.
struct gemm_fetch {
    float d;
    char4 q[4];
    float4 x[2];
};

static inline gemm_fetch gemm_load(thread const gemm_loader & l, uint block) {
    gemm_fetch f;
    device const block_q8_0 * b = l.weight_blocks + block;
    f.d = l.weight_valid ? (float)b->d : 0.0f;
    device const int8_t * q = b->qs + l.load_half * 16;
    for (short i = 0; i < 4; ++i) {
        f.q[i] = char4(*(device const packed_char4 *)(q + 4 * i));
    }

    device const float4 * v = (device const float4 *)(l.x + block * 32);
    f.x[0] = l.token_valid ? v[0] : float4(0.0f);
    f.x[1] = l.token_valid ? v[1] : float4(0.0f);
    return f;
}

// Dequantizes exactly (d * q per element, as the scalar form did) into the threadgroup tile.
static inline void gemm_stage(thread const gemm_loader & l, thread const gemm_fetch & f, threadgroup float * wt,
                              threadgroup float * xt) {
    threadgroup float4 * target = (threadgroup float4 *)(wt + l.load_row * GEMM_K + l.load_half * 16);
    for (short i = 0; i < 4; ++i) {
        target[i] = f.d * float4(f.q[i]);
    }

    threadgroup float4 * xtarget = (threadgroup float4 *)(xt + l.load_token * GEMM_K + l.load_quarter * 8);
    xtarget[0] = f.x[0];
    xtarget[1] = f.x[1];
}

// Simdgroup sg owns rows [32 * (sg % 2), +32) and tokens [16 * (sg / 2), +16) of the tile.
static inline void gemm_accumulate(thread simdgroup_float8x8 (&acc)[8], threadgroup const float * wt,
                                   threadgroup const float * xt, ushort row_base, ushort token_base) {
    for (short k = 0; k < GEMM_K / 8; ++k) {
        simdgroup_float8x8 wm[4];
        simdgroup_float8x8 xm[2];
        for (short i = 0; i < 4; ++i) {
            simdgroup_load(wm[i], wt + (row_base + i * 8) * GEMM_K + k * 8, GEMM_K, ulong2(0, 0), true);
        }

        for (short j = 0; j < 2; ++j) {
            simdgroup_load(xm[j], xt + (token_base + j * 8) * GEMM_K + k * 8, GEMM_K);
        }

        for (short j = 0; j < 2; ++j) {
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(acc[j * 4 + i], xm[j], wm[i], acc[j * 4 + i]);
            }
        }
    }
}

// Writes the staged 32 x 64 result (tokens x rows) with bias, residual, and bounds checks.
static inline void gemm_store(constant matmul_args & a, device const char * weights, device float * output,
                              threadgroup const float * result, uint2 tg, ushort tid) {
    const uint first_row = tg.x * GEMM_ROWS;
    const uint first_token = tg.y * GEMM_TOKENS;
    for (uint idx = tid; idx < GEMM_ROWS * GEMM_TOKENS; idx += 128) {
        const uint t = idx / GEMM_ROWS;
        uint local = first_row + idx % GEMM_ROWS;
        if (local >= a.total_rows || first_token + t >= a.tokens) {
            continue;
        }

        const uint s = locate_segment(a, local);
        float total = result[t * GEMM_ROWS + idx % GEMM_ROWS];
        if (a.segment[s].bias != SYNAPSE_NO_BIAS) {
            total += ((device const float *)(weights + a.segment[s].bias))[local];
        }

        device float * destination = output + (ulong)(first_token + t) * a.out_stride + a.segment[s].out_column + local;
        *destination = a.accumulate != 0 ? *destination + total : total;
    }
}

// Grid: (ceil(total_rows / GEMM_ROWS), ceil(tokens / GEMM_TOKENS)) threadgroups of 128 threads.
kernel void synapse_q8_gemm(
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
    const gemm_loader loader = gemm_prepare(a, weights, input, tg, tid);
    simdgroup_float8x8 acc[8];
    for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const ushort row_base = 32 * (sg % 2);
    const ushort token_base = 16 * (sg / 2);
    const uint blocks = a.columns / 32;
    gemm_fetch next = gemm_load(loader, 0);
    for (uint block = 0; block < blocks; ++block) {
        gemm_stage(loader, next, wt, xt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (block + 1 < blocks) {
            next = gemm_load(loader, block + 1);
        }

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
