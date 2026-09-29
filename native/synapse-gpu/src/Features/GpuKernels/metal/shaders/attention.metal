// Causal grouped-query attention over FP32 or FP16 KV slots with online softmax (flash attention). Each simdgroup
// owns eight (token, query head) rows. Q·K^T and P·V use 8x8 simdgroup matrices with FP32 accumulation. Prompt
// runs and decode tokens both read K/V straight from the slot with little threadgroup memory, so many threadgroups
// share a core; the simdgroups of one KV head hit the same cache lines. Long decode contexts split keys
// across threadgroups and merge in synapse_attention_reduce with the standard max/sum rescaling.

constant constexpr short ATT_KEYS = 32;
constant constexpr short ATT_MAX_SIMDGROUPS = 8;
// Eight-row halves per prompt-run simdgroup (METAL_RUN_TOKENS = 8 x halves on the host). Two halves share each
// K/V load but spill registers on an M2 Pro (32k time to first token 38.1 s against 26.9 s), so one is used.
constant constexpr short ATT_RUN_HALVES = 1;
constant constexpr short ATT_ROW_FLOATS = SYNAPSE_HEAD_DIM + 2;

struct attention_row {
    bool valid;
    uint token;
    uint head;
};

static inline attention_row attention_map(attention_block block, uint group, ushort sg, ushort r) {
    attention_row row;
    if (block.mode == 0) {
        row.valid = sg < group && r < block.token_count;
        row.token = block.first_token + r;
        row.head = block.kv_head * group + sg;
    } else {
        row.valid = sg < block.token_count && r < group;
        row.token = block.first_token + sg;
        row.head = block.kv_head * group + r;
    }

    return row;
}

// Lane coordinates inside an 8x8 simdgroup matrix: thread_elements() holds (row, column) and (row, column + 1).
// This is the fragment layout of Apple GPUs that MLX's steel kernels also rely on; the Metal tests pin the
// result against the scalar reference. Lanes that share a row differ only in bits 0 and 3.
static inline ushort fragment_row(ushort lane) {
    return ((lane >> 2) & 4) + ((lane >> 1) & 3);
}

static inline ushort fragment_column(ushort lane) {
    return ((lane >> 2) & 2) * 2 + (lane & 1) * 2;
}

// The two elements this lane physically holds. thread_elements() is typed as the whole 8x8 vector, so it is
// reinterpreted as the lane's pair (as MLX does); indexing the 64-element type directly makes the compiler spill.
static inline thread float2 & lane_pair(thread simdgroup_float8x8 & matrix) {
    return reinterpret_cast<thread float2 &>(matrix.thread_elements());
}

// One simdgroup's online-softmax state over its eight rows; each lane tracks the row it holds in every fragment.
struct attention_state {
    float2 out[8];
    float running_max;
    float running_sum;
};

// Scales and masks one half's scores, updates its running max and sum, and replaces the scores with
// probabilities. Returns the factor that rescales the half's earlier output.
static inline float softmax_scores(
        thread attention_state & state,
        thread float2 (&scores)[ATT_KEYS / 8],
        uint first_key,
        uint count,
        bool row_valid,
        uint row_position,
        float scale,
        ushort lane) {
    const ushort column = fragment_column(lane);
    float tile_max = -INFINITY;
    for (short c = 0; c < ATT_KEYS / 8; ++c) {
        for (short t = 0; t < 2; ++t) {
            const uint local = c * 8 + column + t;
            const bool visible = row_valid && local < count && first_key + local <= row_position;
            scores[c][t] = visible ? scores[c][t] * scale : -INFINITY;
            tile_max = max(tile_max, scores[c][t]);
        }
    }

    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
    const float next_max = max(state.running_max, tile_max);
    const float alpha = next_max == -INFINITY ? 1.0f : exp(state.running_max - next_max);
    float tile_sum = 0.0f;
    for (short c = 0; c < ATT_KEYS / 8; ++c) {
        for (short t = 0; t < 2; ++t) {
            scores[c][t] = scores[c][t] == -INFINITY ? 0.0f : exp(scores[c][t] - next_max);
            tile_sum += scores[c][t];
        }
    }

    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 8);
    state.running_sum = state.running_sum * alpha + tile_sum;
    state.running_max = next_max;
    return alpha;
}

// Folds one tile of up to ATT_KEYS keys (row-major [key][64] at `keys` and `values`) into `H` halves of eight
// query rows each. Every key and value fragment is loaded once and multiplied into all halves, so more rows share
// each load. Scores and outputs stay in registers as each lane's element pair; matrices exist only per multiply.
// `T` is the element type the tile is read as; half tiles multiply into FP32.
template <typename T, short H, typename Pointer>
static inline void attend_tile(
        thread attention_state (&state)[H],
        thread const simdgroup_float8x8 (&query)[H][8],
        Pointer keys,
        Pointer values,
        uint first_key,
        uint count,
        thread const bool (&row_valid)[H],
        thread const uint (&row_position)[H],
        float scale,
        ushort lane) {
    float2 scores[H][ATT_KEYS / 8];
    for (short c = 0; c < ATT_KEYS / 8; ++c) {
        simdgroup_float8x8 product[H];
        for (short h = 0; h < H; ++h) {
            product[h] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }

        for (short dk = 0; dk < 8; ++dk) {
            simdgroup_matrix<T, 8, 8> key_block;
            simdgroup_load(key_block, keys + c * 8 * SYNAPSE_HEAD_DIM + dk * 8, SYNAPSE_HEAD_DIM, ulong2(0, 0), true);
            for (short h = 0; h < H; ++h) {
                simdgroup_multiply_accumulate(product[h], query[h][dk], key_block, product[h]);
            }
        }

        for (short h = 0; h < H; ++h) {
            scores[h][c] = lane_pair(product[h]);
        }
    }

    float alpha[H];
    for (short h = 0; h < H; ++h) {
        alpha[h] = softmax_scores(state[h], scores[h], first_key, count, row_valid[h], row_position[h], scale, lane);
    }

    for (short j = 0; j < 8; ++j) {
        simdgroup_float8x8 output[H];
        for (short h = 0; h < H; ++h) {
            lane_pair(output[h]) = state[h].out[j] * alpha[h];
        }

        for (short k = 0; k < ATT_KEYS / 8; ++k) {
            simdgroup_matrix<T, 8, 8> value_block;
            simdgroup_load(value_block, values + k * 8 * SYNAPSE_HEAD_DIM + j * 8, SYNAPSE_HEAD_DIM);
            for (short h = 0; h < H; ++h) {
                simdgroup_float8x8 probability;
                lane_pair(probability) = scores[h][k];
                simdgroup_multiply_accumulate(output[h], probability, value_block, output[h]);
            }
        }

        for (short h = 0; h < H; ++h) {
            state[h].out[j] = lane_pair(output[h]);
        }
    }
}

// Writes a simdgroup's rows: normalized output for one split, or unnormalized output plus max and sum.
static inline void attention_finish(
        constant attention_args & a,
        thread attention_state & state,
        attention_block block,
        threadgroup float * stage,
        threadgroup float * row_state,
        device float * output,
        device float * partial,
        uint block_index,
        uint split,
        ushort sg,
        ushort lane,
        ushort row_base) {
    for (short j = 0; j < 8; ++j) {
        simdgroup_float8x8 output;
        lane_pair(output) = state.out[j];
        simdgroup_store(output, stage + j * 8, SYNAPSE_HEAD_DIM);
    }

    if (fragment_column(lane) == 0) {
        row_state[fragment_row(lane)] = state.running_sum;
        row_state[8 + fragment_row(lane)] = state.running_max;
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort idx = lane; idx < 8 * SYNAPSE_HEAD_DIM; idx += 32) {
        const ushort r = idx / SYNAPSE_HEAD_DIM;
        const ushort d = idx % SYNAPSE_HEAD_DIM;
        const attention_row row = attention_map(block, a.group, sg, row_base + r);
        if (!row.valid) {
            continue;
        }

        if (a.splits == 1) {
            output[(ulong)row.token * a.out_stride + row.head * SYNAPSE_HEAD_DIM + d] = stage[idx] / row_state[r];
            continue;
        }

        device float * target = partial + (((ulong)block_index * a.splits + split) * 64 + sg * 8 + r) * ATT_ROW_FLOATS;
        target[d] = stage[idx];
        if (d == 0) {
            target[SYNAPSE_HEAD_DIM] = row_state[8 + r];
            target[SYNAPSE_HEAD_DIM + 1] = row_state[r];
        }
    }
}

// Stages eight 64-wide query rows starting at `row_base` (zero for invalid rows) and loads them as fragments.
static inline void stage_query(constant attention_args & a, attention_block block, device const float * qkv,
                               threadgroup float * stage, ushort sg, ushort lane, ushort row_base,
                               thread simdgroup_float8x8 (&query)[8]) {
    for (ushort idx = lane; idx < 8 * SYNAPSE_HEAD_DIM; idx += 32) {
        const attention_row row = attention_map(block, a.group, sg, row_base + idx / SYNAPSE_HEAD_DIM);
        stage[idx] = row.valid
            ? qkv[(ulong)row.token * a.qkv_stride + row.head * SYNAPSE_HEAD_DIM + idx % SYNAPSE_HEAD_DIM]
            : 0.0f;
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (short dk = 0; dk < 8; ++dk) {
        simdgroup_load(query[dk], stage + dk * 8, SYNAPSE_HEAD_DIM);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
}

static inline attention_state attention_start() {
    attention_state state;
    for (short j = 0; j < 8; ++j) {
        state.out[j] = float2(0.0f);
    }

    state.running_max = -INFINITY;
    state.running_sum = 0.0f;
    return state;
}

// Prompt runs (mode 0). Grid: (run blocks, 1). Threadgroup: group * 32 threads, one simdgroup per query head.
// Every simdgroup reads K/V straight from the slot (the group's simdgroups hit the same cache lines), keeps its
// scores in registers, and owns a private threadgroup region for query and output staging. No threadgroup barrier
// is needed, and the small footprint lets several threadgroups share a core.
template <typename KV>
kernel void synapse_attention(
        constant attention_args & a [[buffer(0)]],
        device const float * qkv [[buffer(1)]],
        device float * output [[buffer(2)]],
        device const batch_token * tokens [[buffer(3)]],
        constant slot_table & slots [[buffer(4)]],
        device const attention_block * blocks [[buffer(5)]],
        device float * partial [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort2 threads_2d [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float region[ATT_MAX_SIMDGROUPS][8 * SYNAPSE_HEAD_DIM];
    threadgroup float row_state[ATT_MAX_SIMDGROUPS][16];
    if (sg >= a.group) {
        return;
    }

    const attention_block block = blocks[a.block_base + tg.x];
    const uint split = tg.y;
    const uint key_begin = min(block.key_end, split * a.split_keys);
    const uint key_end = min(block.key_end, key_begin + a.split_keys);
    device const KV * kv = (device const KV *)slots.address[block.slot];
    const uint capacity = slots.capacity[block.slot];
    const ulong key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, capacity);
    const ulong value_base = kv_value_base(a.layers, a.kv_heads, capacity) + key_base;

    threadgroup float * stage = region[sg];
    simdgroup_float8x8 query[ATT_RUN_HALVES][8];
    attention_state state[ATT_RUN_HALVES];
    bool valid[ATT_RUN_HALVES];
    uint position[ATT_RUN_HALVES];
    for (short h = 0; h < ATT_RUN_HALVES; ++h) {
        stage_query(a, block, qkv, stage, sg, lane, h * 8, query[h]);
        state[h] = attention_start();
        const attention_row mine = attention_map(block, a.group, sg, h * 8 + fragment_row(lane));
        valid[h] = mine.valid;
        position[h] = mine.valid ? (uint)tokens[mine.token].position : 0;
    }

    for (uint first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const uint count = min((uint)ATT_KEYS, key_end - first_key);
        const ulong offset = (ulong)first_key * SYNAPSE_HEAD_DIM;
        attend_tile<KV, ATT_RUN_HALVES>(state, query, kv + key_base + offset, kv + value_base + offset, first_key, count,
                                        valid, position, a.scale, lane);
    }

    for (short h = 0; h < ATT_RUN_HALVES; ++h) {
        attention_finish(a, state[h], block, stage, row_state[sg], output, partial, tg.x, split, sg, lane, h * 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    (void)tid;
    (void)threads_2d;
}

// Decode tokens (mode 1). Grid: (decode blocks, splits). Threadgroup: one simdgroup whose rows are the
// query heads of one KV head. K/V come straight from the slot; a partial last tile reads stale but finite
// slot memory (the slot has one tile of padding), and those keys are masked.
template <typename KV>
kernel void synapse_attention_decode(
        constant attention_args & a [[buffer(0)]],
        device const float * qkv [[buffer(1)]],
        device float * output [[buffer(2)]],
        device const batch_token * tokens [[buffer(3)]],
        constant slot_table & slots [[buffer(4)]],
        device const attention_block * blocks [[buffer(5)]],
        device float * partial [[buffer(6)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float stage[8 * SYNAPSE_HEAD_DIM];
    threadgroup float row_state[16];

    const attention_block block = blocks[a.block_base + tg.x];
    const uint split = tg.y;
    const uint key_begin = min(block.key_end, split * a.split_keys);
    const uint key_end = min(block.key_end, key_begin + a.split_keys);
    device const KV * kv = (device const KV *)slots.address[block.slot];
    const uint capacity = slots.capacity[block.slot];
    const ulong key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, capacity);
    const ulong value_base = kv_value_base(a.layers, a.kv_heads, capacity) + key_base;

    simdgroup_float8x8 query[1][8];
    stage_query(a, block, qkv, stage, 0, lane, 0, query[0]);
    attention_state state[1] = {attention_start()};
    const attention_row mine = attention_map(block, a.group, 0, fragment_row(lane));
    const bool valid[1] = {mine.valid};
    const uint position[1] = {mine.valid ? (uint)tokens[mine.token].position : 0};
    for (uint first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const uint count = min((uint)ATT_KEYS, key_end - first_key);
        const ulong offset = (ulong)first_key * SYNAPSE_HEAD_DIM;
        attend_tile<KV, 1>(state, query, kv + key_base + offset, kv + value_base + offset, first_key, count, valid,
                           position, a.scale, lane);
    }

    attention_finish(a, state[0], block, stage, row_state, output, partial, tg.x, split, 0, lane, 0);
}

// Grid (64 dims, 64 rows, blocks): merges the splits of one decode dispatch into the attention output.
kernel void synapse_attention_reduce(
        constant attention_args & a [[buffer(0)]],
        device const attention_block * blocks [[buffer(1)]],
        device const float * partial [[buffer(2)]],
        device float * output [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row_index = gid.y;
    const attention_block block = blocks[a.block_base + gid.z];
    const attention_row row = attention_map(block, a.group, row_index / 8, row_index % 8);
    if (d >= SYNAPSE_HEAD_DIM || !row.valid) {
        return;
    }

    float maximum = -INFINITY;
    for (uint split = 0; split < a.splits; ++split) {
        device const float * source = partial + (((ulong)gid.z * a.splits + split) * 64 + row_index) * ATT_ROW_FLOATS;
        if (source[SYNAPSE_HEAD_DIM + 1] > 0.0f) {
            maximum = max(maximum, source[SYNAPSE_HEAD_DIM]);
        }
    }

    float sum = 0.0f;
    float value = 0.0f;
    for (uint split = 0; split < a.splits; ++split) {
        device const float * source = partial + (((ulong)gid.z * a.splits + split) * 64 + row_index) * ATT_ROW_FLOATS;
        const float weight_sum = source[SYNAPSE_HEAD_DIM + 1];
        if (weight_sum > 0.0f) {
            const float factor = exp(source[SYNAPSE_HEAD_DIM] - maximum);
            sum += weight_sum * factor;
            value += source[d] * factor;
        }
    }

    output[(ulong)row.token * a.out_stride + row.head * SYNAPSE_HEAD_DIM + d] = value / sum;
}

#define SYNAPSE_ATTENTION(NAME, KERNEL, KV) \
    template [[host_name(NAME)]] kernel void KERNEL<KV>( \
        constant attention_args &, device const float *, device float *, device const batch_token *, \
        constant slot_table &, device const attention_block *, device float *, uint2, ushort, ushort2, ushort, \
        ushort);

SYNAPSE_ATTENTION("synapse_attention", synapse_attention, float)
SYNAPSE_ATTENTION("synapse_attention_f16", synapse_attention, half)

#define SYNAPSE_ATTENTION_DECODE(NAME, KV) \
    template [[host_name(NAME)]] kernel void synapse_attention_decode<KV>( \
        constant attention_args &, device const float *, device float *, device const batch_token *, \
        constant slot_table &, device const attention_block *, device float *, uint2, ushort);

SYNAPSE_ATTENTION_DECODE("synapse_attention_decode", float)
SYNAPSE_ATTENTION_DECODE("synapse_attention_decode_f16", half)
