// Causal grouped-query flash attention over FP32 KV slots (same semantics and geometry as attention.metal).
// A warp owns eight (token, query head) rows; lane j scores key j of each 32-key tile and owns output
// dimensions 2 * lane and 2 * lane + 1.

#define ATT_KEYS 32
#define ATT_ROW_FLOATS (SYNAPSE_HEAD_DIM + 2)

struct attention_row {
    bool valid;
    u32 token;
    u32 head;
};

__device__ __forceinline__ attention_row attention_map(const attention_block & block, u32 group, u32 warp, u32 r) {
    attention_row row;
    if (block.mode == 0) {
        row.valid = warp < group && r < block.token_count;
        row.token = block.first_token + r;
        row.head = block.kv_head * group + warp;
    } else {
        row.valid = warp < block.token_count && r < group;
        row.token = block.first_token + warp;
        row.head = block.kv_head * group + r;
    }

    return row;
}

struct attention_state {
    float out[8][2];
    float running_max[8];
    float running_sum[8];
};

// Folds up to ATT_KEYS keys into the warp's state. `keys` and `values` are generic pointers (shared or
// global) to row-major rows with the given strides.
__device__ void attend_tile(
        attention_state & state, const float (*query)[SYNAPSE_HEAD_DIM], const float * keys, u32 key_stride,
        const float * values, u32 value_stride, u32 first_key, u32 count, const bool * row_valid,
        const u32 * row_position, float scale, u32 lane) {
    float score[8];
    for (int r = 0; r < 8; ++r) {
        score[r] = 0.0f;
    }

    if (lane < count) {
        const float * key = keys + (u64)lane * key_stride;
        for (int d = 0; d < SYNAPSE_HEAD_DIM; ++d) {
            const float k = key[d];
            for (int r = 0; r < 8; ++r) {
                score[r] = fmaf(query[r][d], k, score[r]);
            }
        }
    }

    float probability[8];
    float alpha[8];
    for (int r = 0; r < 8; ++r) {
        const u32 key = first_key + lane;
        const bool visible = row_valid[r] && lane < count && key <= row_position[r];
        const float s = visible ? score[r] * scale : -INFINITY;
        const float next_max = fmaxf(state.running_max[r], warp_max(s));
        alpha[r] = next_max == -INFINITY ? 1.0f : expf(state.running_max[r] - next_max);
        probability[r] = s == -INFINITY ? 0.0f : expf(s - next_max);
        state.running_sum[r] = state.running_sum[r] * alpha[r] + warp_sum(probability[r]);
        state.running_max[r] = next_max;
        state.out[r][0] *= alpha[r];
        state.out[r][1] *= alpha[r];
    }

    for (u32 j = 0; j < ATT_KEYS; ++j) {
        const float2 value = j < count
            ? *(const float2 *)(values + (u64)j * value_stride + 2 * lane)
            : make_float2(0.0f, 0.0f);
        for (int r = 0; r < 8; ++r) {
            const float p = __shfl_sync(SYNAPSE_FULL_MASK, probability[r], j);
            state.out[r][0] = fmaf(p, value.x, state.out[r][0]);
            state.out[r][1] = fmaf(p, value.y, state.out[r][1]);
        }
    }
}

__device__ void attention_finish(
        const attention_args & a, const attention_state & state, const attention_block & block, u32 warp,
        u32 lane, u32 block_index, u32 split, float * output, float * partial) {
    for (u32 r = 0; r < 8; ++r) {
        const attention_row row = attention_map(block, a.group, warp, r);
        if (!row.valid) {
            continue;
        }

        if (a.splits == 1) {
            float * target = output + (u64)row.token * a.out_stride + row.head * SYNAPSE_HEAD_DIM + 2 * lane;
            target[0] = state.out[r][0] / state.running_sum[r];
            target[1] = state.out[r][1] / state.running_sum[r];
            continue;
        }

        float * target = partial + (((u64)block_index * a.splits + split) * 64 + warp * 8 + r) * ATT_ROW_FLOATS;
        target[2 * lane] = state.out[r][0];
        target[2 * lane + 1] = state.out[r][1];
        if (lane == 0) {
            target[SYNAPSE_HEAD_DIM] = state.running_max[r];
            target[SYNAPSE_HEAD_DIM + 1] = state.running_sum[r];
        }
    }
}

__device__ void attention_init(attention_state & state) {
    for (int r = 0; r < 8; ++r) {
        state.out[r][0] = 0.0f;
        state.out[r][1] = 0.0f;
        state.running_max[r] = -INFINITY;
        state.running_sum[r] = 0.0f;
    }
}

// Prompt runs (mode 0). Grid (run blocks, 1) of group * 32 threads; the K/V tile is staged in shared memory.
extern "C" __global__ void synapse_attention(
        attention_args a, const float * qkv, float * output, const batch_token * tokens, slot_table slots,
        const attention_block * blocks, float * partial) {
    __shared__ float keys[ATT_KEYS][SYNAPSE_HEAD_DIM + 1];
    __shared__ float values[ATT_KEYS][SYNAPSE_HEAD_DIM];
    __shared__ float query[8][8][SYNAPSE_HEAD_DIM];
    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    const attention_block block = blocks[a.block_base + blockIdx.x];
    const u32 split = blockIdx.y;
    const u32 key_begin = min(block.key_end, split * a.split_keys);
    const u32 key_end = min(block.key_end, key_begin + a.split_keys);
    const float * kv = slots.address[block.slot];
    const u64 key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, a.context);
    const u64 value_base = kv_value_base(a.layers, a.kv_heads, a.context) + key_base;

    bool row_valid[8];
    u32 row_position[8];
    for (u32 r = 0; r < 8; ++r) {
        const attention_row row = attention_map(block, a.group, warp, r);
        row_valid[r] = row.valid;
        row_position[r] = row.valid ? (u32)tokens[row.token].position : 0;
        for (u32 d = lane; d < SYNAPSE_HEAD_DIM; d += 32) {
            query[warp][r][d] = row.valid ? qkv[(u64)row.token * a.qkv_stride + row.head * SYNAPSE_HEAD_DIM + d] : 0.0f;
        }
    }

    attention_state state;
    attention_init(state);
    for (u32 first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const u32 count = min((u32)ATT_KEYS, key_end - first_key);
        __syncthreads();
        for (u32 idx = threadIdx.x; idx < ATT_KEYS * SYNAPSE_HEAD_DIM; idx += blockDim.x) {
            const u32 key = idx / SYNAPSE_HEAD_DIM;
            const u32 d = idx % SYNAPSE_HEAD_DIM;
            const u64 source = (u64)(first_key + key) * SYNAPSE_HEAD_DIM + d;
            keys[key][d] = key < count ? kv[key_base + source] : 0.0f;
            values[key][d] = key < count ? kv[value_base + source] : 0.0f;
        }

        __syncthreads();
        if (warp < a.group) {
            attend_tile(state, query[warp], &keys[0][0], SYNAPSE_HEAD_DIM + 1, &values[0][0], SYNAPSE_HEAD_DIM,
                        first_key, count, row_valid, row_position, a.scale, lane);
        }
    }

    if (warp < a.group) {
        attention_finish(a, state, block, warp, lane, blockIdx.x, split, output, partial);
    }
}

// Decode tokens (mode 1). Grid (decode blocks, splits) of 32 threads; K/V are read straight from the slot.
extern "C" __global__ void synapse_attention_decode(
        attention_args a, const float * qkv, float * output, const batch_token * tokens, slot_table slots,
        const attention_block * blocks, float * partial) {
    __shared__ float query[1][8][SYNAPSE_HEAD_DIM];
    const u32 lane = threadIdx.x & 31;
    const attention_block block = blocks[a.block_base + blockIdx.x];
    const u32 split = blockIdx.y;
    const u32 key_begin = min(block.key_end, split * a.split_keys);
    const u32 key_end = min(block.key_end, key_begin + a.split_keys);
    const float * kv = slots.address[block.slot];
    const u64 key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, a.context);
    const u64 value_base = kv_value_base(a.layers, a.kv_heads, a.context) + key_base;

    bool row_valid[8];
    u32 row_position[8];
    for (u32 r = 0; r < 8; ++r) {
        const attention_row row = attention_map(block, a.group, 0, r);
        row_valid[r] = row.valid;
        row_position[r] = row.valid ? (u32)tokens[row.token].position : 0;
        for (u32 d = lane; d < SYNAPSE_HEAD_DIM; d += 32) {
            query[0][r][d] = row.valid ? qkv[(u64)row.token * a.qkv_stride + row.head * SYNAPSE_HEAD_DIM + d] : 0.0f;
        }
    }

    __syncwarp();
    attention_state state;
    attention_init(state);
    for (u32 first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const u32 count = min((u32)ATT_KEYS, key_end - first_key);
        const u64 offset = (u64)first_key * SYNAPSE_HEAD_DIM;
        attend_tile(state, query[0], kv + key_base + offset, SYNAPSE_HEAD_DIM, kv + value_base + offset,
                    SYNAPSE_HEAD_DIM, first_key, count, row_valid, row_position, a.scale, lane);
    }

    attention_finish(a, state, block, 0, lane, blockIdx.x, split, output, partial);
}

// Grid (1, 64 rows, blocks) of 64 threads: merges the splits of one decode dispatch.
extern "C" __global__ void synapse_attention_reduce(
        attention_args a, const attention_block * blocks, const float * partial, float * output) {
    const u32 d = threadIdx.x;
    const u32 row_index = blockIdx.y;
    const attention_block block = blocks[a.block_base + blockIdx.z];
    const attention_row row = attention_map(block, a.group, row_index / 8, row_index % 8);
    if (d >= SYNAPSE_HEAD_DIM || !row.valid) {
        return;
    }

    float maximum = -INFINITY;
    for (u32 split = 0; split < a.splits; ++split) {
        const float * source = partial + (((u64)blockIdx.z * a.splits + split) * 64 + row_index) * ATT_ROW_FLOATS;
        if (source[SYNAPSE_HEAD_DIM + 1] > 0.0f) {
            maximum = fmaxf(maximum, source[SYNAPSE_HEAD_DIM]);
        }
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (u32 split = 0; split < a.splits; ++split) {
        const float * source = partial + (((u64)blockIdx.z * a.splits + split) * 64 + row_index) * ATT_ROW_FLOATS;
        const float weight_sum = source[SYNAPSE_HEAD_DIM + 1];
        if (weight_sum > 0.0f) {
            const float factor = expf(source[SYNAPSE_HEAD_DIM] - maximum);
            sum += weight_sum * factor;
            value += source[d] * factor;
        }
    }

    output[(u64)row.token * a.out_stride + row.head * SYNAPSE_HEAD_DIM + d] = value / sum;
}
