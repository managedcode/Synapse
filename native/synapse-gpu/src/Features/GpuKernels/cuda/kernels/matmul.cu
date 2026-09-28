// Q8_0 matrix times FP32 token rows (same numeric contract and geometry as matmul.metal): a batch-invariant
// matrix-vector kernel with a fixed fma order, and a shared-memory tiled GEMM for prompt runs.

#define MV_ROWS 2
#define MV_WARPS 4

__device__ __forceinline__ float dot8(const signed char * q, float4 x0, float4 x1) {
    float acc = (float)q[0] * x0.x;
    acc = fmaf((float)q[1], x0.y, acc);
    acc = fmaf((float)q[2], x0.z, acc);
    acc = fmaf((float)q[3], x0.w, acc);
    acc = fmaf((float)q[4], x1.x, acc);
    acc = fmaf((float)q[5], x1.y, acc);
    acc = fmaf((float)q[6], x1.z, acc);
    acc = fmaf((float)q[7], x1.w, acc);
    return acc;
}

struct matvec_rows {
    const block_q8_0 * row[MV_ROWS];
    u32 out_column[MV_ROWS];
    u32 local_row[MV_ROWS];
    u64 bias[MV_ROWS];
    bool valid[MV_ROWS];
};

__device__ __forceinline__ matvec_rows matvec_locate(const matmul_args & a, const char * weights, u32 first_row) {
    const u64 row_bytes = (u64)(a.columns / 32) * sizeof(block_q8_0);
    matvec_rows rows;
    for (int r = 0; r < MV_ROWS; ++r) {
        const u32 row = first_row + r;
        rows.valid[r] = row < a.total_rows;
        u32 local = rows.valid[r] ? row : 0;
        const u32 s = locate_segment(a, local);
        rows.row[r] = (const block_q8_0 *)(weights + a.segment[s].weight + (u64)local * row_bytes);
        rows.out_column[r] = a.segment[s].out_column + local;
        rows.local_row[r] = local;
        rows.bias[r] = a.segment[s].bias;
    }

    return rows;
}

// Reduces the lane sums across the warps in a fixed order and writes each row with bias and residual.
template <int T>
__device__ void matvec_finish(const matmul_args & a, const char * weights, float * output, const matvec_rows & rows,
                              float (&sums)[MV_ROWS][T], float (&partial)[MV_WARPS][MV_ROWS][T], u32 first_token,
                              u32 tokens) {
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    for (int r = 0; r < MV_ROWS; ++r) {
        for (int t = 0; t < T; ++t) {
            const float total = warp_sum(sums[r][t]);
            if (lane == 0) {
                partial[warp][r][t] = total;
            }
        }
    }

    __syncthreads();
    const u32 id = threadIdx.x;
    const u32 r = id / T;
    const u32 t = id % T;
    if (id >= MV_ROWS * T || !rows.valid[r] || t >= tokens) {
        return;
    }

    float total = ((partial[0][r][t] + partial[1][r][t]) + partial[2][r][t]) + partial[3][r][t];
    if (rows.bias[r] != SYNAPSE_NO_BIAS) {
        total += ((const float *)(weights + rows.bias[r]))[rows.local_row[r]];
    }

    float * destination = output + (u64)(first_token + t) * a.out_stride + rows.out_column[r];
    *destination = a.accumulate != 0 ? *destination + total : total;
}

// Grid (ceil(total_rows / 2), ceil(tokens / T)) of 128 threads.
template <int T>
__device__ void q8_matvec(matmul_args a, const char * weights, const float * input, float * output) {
    __shared__ float partial[MV_WARPS][MV_ROWS][T];
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    const u32 first_token = blockIdx.y * T;
    const u32 tokens = min((u32)T, a.tokens - first_token);
    const u32 blocks = a.columns / 32;
    const matvec_rows rows = matvec_locate(a, weights, blockIdx.x * MV_ROWS);
    float sums[MV_ROWS][T] = {};
    const u32 quarter = lane % 4;
    const float * x = input + (u64)first_token * a.in_stride + quarter * 8;
    for (u32 block = warp * 8 + lane / 4; block < blocks; block += MV_WARPS * 8) {
        float4 x0[T];
        float4 x1[T];
        for (int t = 0; t < T; ++t) {
            const float4 * v = (const float4 *)(x + (u64)min((u32)t, tokens - 1) * a.in_stride + block * 32);
            x0[t] = v[0];
            x1[t] = v[1];
        }

        for (int r = 0; r < MV_ROWS; ++r) {
            const block_q8_0 * b = rows.row[r] + (rows.valid[r] ? block : 0);
            const float d = half_to_float(b->d);
            const signed char * q = b->qs + quarter * 8;
            for (int t = 0; t < T; ++t) {
                sums[r][t] = fmaf(dot8(q, x0[t], x1[t]), d, sums[r][t]);
            }
        }
    }

    matvec_finish<T>(a, weights, output, rows, sums, partial, first_token, tokens);
}

extern "C" __global__ void synapse_q8_matvec_1(matmul_args a, const char * w, const float * x, float * y) { q8_matvec<1>(a, w, x, y); }
extern "C" __global__ void synapse_q8_matvec_2(matmul_args a, const char * w, const float * x, float * y) { q8_matvec<2>(a, w, x, y); }
extern "C" __global__ void synapse_q8_matvec_4(matmul_args a, const char * w, const float * x, float * y) { q8_matvec<4>(a, w, x, y); }
extern "C" __global__ void synapse_q8_matvec_8(matmul_args a, const char * w, const float * x, float * y) { q8_matvec<8>(a, w, x, y); }

#define GEMM_ROWS 64
#define GEMM_TOKENS 32
#define GEMM_K 32

// Writes one thread's 4 rows x 4 tokens with bias, residual, and bounds checks.
__device__ void gemm_store(const matmul_args & a, const char * weights, float * output, float (&acc)[4][4],
                           u32 first_row, u32 first_token) {
    for (int i = 0; i < 4; ++i) {
        u32 local = first_row + i;
        if (local >= a.total_rows) {
            continue;
        }

        const u32 s = locate_segment(a, local);
        const u64 bias_offset = a.segment[s].bias;
        const float bias = bias_offset != SYNAPSE_NO_BIAS ? ((const float *)(weights + bias_offset))[local] : 0.0f;
        for (int j = 0; j < 4; ++j) {
            const u32 token = first_token + j;
            if (token >= a.tokens) {
                continue;
            }

            float * destination = output + (u64)token * a.out_stride + a.segment[s].out_column + local;
            const float total = acc[i][j] + bias;
            *destination = a.accumulate != 0 ? *destination + total : total;
        }
    }
}

// Grid (ceil(total_rows / 64), ceil(tokens / 32)) of 128 threads; each thread owns 4 rows x 4 tokens.
extern "C" __global__ void synapse_q8_gemm(matmul_args a, const char * weights, const float * input, float * output) {
    __shared__ float wt[GEMM_ROWS][GEMM_K + 1];
    __shared__ float xt[GEMM_TOKENS][GEMM_K + 1];
    const u32 tid = threadIdx.x;
    const u32 first_row = blockIdx.x * GEMM_ROWS;
    const u32 first_token = blockIdx.y * GEMM_TOKENS;
    const u32 blocks = a.columns / 32;
    const bool weight_valid = first_row + tid / 2 < a.total_rows;
    u32 local = weight_valid ? first_row + tid / 2 : 0;
    const u32 segment = locate_segment(a, local);
    const block_q8_0 * weight_blocks =
        (const block_q8_0 *)(weights + a.segment[segment].weight + (u64)local * (blocks * sizeof(block_q8_0)));
    const bool token_valid = first_token + tid / 4 < a.tokens;
    const float * x = input + (u64)(token_valid ? first_token + tid / 4 : 0) * a.in_stride + (tid % 4) * 8;
    const u32 row_base = (tid % 16) * 4;
    const u32 token_base = (tid / 16) * 4;
    float acc[4][4] = {};
    for (u32 block = 0; block < blocks; ++block) {
        const block_q8_0 * b = weight_blocks + block;
        const float d = weight_valid ? half_to_float(b->d) : 0.0f;
        for (int i = 0; i < 16; ++i) {
            wt[tid / 2][(tid % 2) * 16 + i] = d * (float)b->qs[(tid % 2) * 16 + i];
        }

        for (int i = 0; i < 8; ++i) {
            xt[tid / 4][(tid % 4) * 8 + i] = token_valid ? x[block * 32 + i] : 0.0f;
        }

        __syncthreads();
        for (int k = 0; k < GEMM_K; ++k) {
            for (int j = 0; j < 4; ++j) {
                const float xv = xt[token_base + j][k];
                for (int i = 0; i < 4; ++i) {
                    acc[i][j] = fmaf(wt[row_base + i][k], xv, acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    gemm_store(a, weights, output, acc, first_row + row_base, first_token + token_base);
}
