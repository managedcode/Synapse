// Causal grouped-query attention over FP32 or FP16 KV slots with online softmax (flash attention). Each simdgroup
// owns eight (token, query head) rows. Q·K^T and P·V use 8x8 simdgroup matrices with FP32 accumulation. Prompt
// runs and decode tokens both read K/V straight from the slot with little threadgroup memory, so many threadgroups
// share a core; the simdgroups of one KV head hit the same cache lines. Long decode contexts split keys
// across threadgroups and merge in synapse_attention_reduce with the standard max/sum rescaling.

constant constexpr short ATT_KEYS = 32;
constant constexpr short ATT_MAX_SIMDGROUPS = 8;
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

// One simdgroup's online-softmax state over its eight rows. Lane l keeps row l / 4 and columns
// (l % 4) * 8 .. +8 of each score tile.
struct attention_state {
    simdgroup_float8x8 out[8];
    float running_max;
    float running_sum;
};

// Masks one 8 x ATT_KEYS score tile, updates each row's running max and sum, rewrites the scores as
// probabilities, and writes the diagonal rescale matrix. Lane l owns row l / 4, columns (l % 4) * 8 .. +8.
static inline void online_softmax(
        thread attention_state & state,
        threadgroup float * scores,
        threadgroup float * diagonal,
        uint first_key,
        uint count,
        bool row_valid,
        uint row_position,
        float scale,
        ushort lane) {
    const ushort my_row = lane / 4;
    const ushort my_column = (lane % 4) * 8;
    float local[8];
    float local_max = -INFINITY;
    for (short j = 0; j < 8; ++j) {
        const uint key = first_key + my_column + j;
        const bool visible = row_valid && (uint)(my_column + j) < count && key <= row_position;
        local[j] = visible ? scores[my_row * ATT_KEYS + my_column + j] * scale : -INFINITY;
        local_max = max(local_max, local[j]);
    }

    local_max = max(local_max, simd_shuffle_xor(local_max, 1));
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    const float next_max = max(state.running_max, local_max);
    const float alpha = next_max == -INFINITY ? 1.0f : exp(state.running_max - next_max);
    float local_sum = 0.0f;
    for (short j = 0; j < 8; ++j) {
        const float p = local[j] == -INFINITY ? 0.0f : exp(local[j] - next_max);
        scores[my_row * ATT_KEYS + my_column + j] = p;
        local_sum += p;
    }

    local_sum += simd_shuffle_xor(local_sum, 1);
    local_sum += simd_shuffle_xor(local_sum, 2);
    state.running_sum = state.running_sum * alpha + local_sum;
    state.running_max = next_max;
    if (lane % 4 == 0) {
        for (short j = 0; j < 8; ++j) {
            diagonal[my_row * 8 + j] = j == (short)my_row ? alpha : 0.0f;
        }
    }
}

// Folds one tile of up to ATT_KEYS keys (row-major [key][64] at `keys` and `values`) into `state`. `T` is the
// element type the tile is read as; half tiles multiply into FP32 accumulators.
template <typename T, typename Pointer>
static inline void attend_tile(
        thread attention_state & state,
        thread const simdgroup_float8x8 (&query)[8],
        Pointer keys,
        Pointer values,
        threadgroup float * scores,
        threadgroup float * diagonal,
        uint first_key,
        uint count,
        bool row_valid,
        uint row_position,
        float scale,
        ushort lane) {
    for (short c = 0; c < ATT_KEYS / 8; ++c) {
        simdgroup_float8x8 score = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (short dk = 0; dk < 8; ++dk) {
            simdgroup_matrix<T, 8, 8> key_block;
            simdgroup_load(key_block, keys + c * 8 * SYNAPSE_HEAD_DIM + dk * 8, SYNAPSE_HEAD_DIM, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(score, query[dk], key_block, score);
        }

        simdgroup_store(score, scores + c * 8, ATT_KEYS);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    online_softmax(state, scores, diagonal, first_key, count, row_valid, row_position, scale, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 rescale;
    simdgroup_load(rescale, diagonal, 8);
    for (short j = 0; j < 8; ++j) {
        simdgroup_multiply(state.out[j], rescale, state.out[j]);
    }

    for (short k = 0; k < ATT_KEYS / 8; ++k) {
        simdgroup_float8x8 probability;
        simdgroup_load(probability, scores + k * 8, ATT_KEYS);
        for (short j = 0; j < 8; ++j) {
            simdgroup_matrix<T, 8, 8> value_block;
            simdgroup_load(value_block, values + k * 8 * SYNAPSE_HEAD_DIM + j * 8, SYNAPSE_HEAD_DIM);
            simdgroup_multiply_accumulate(state.out[j], probability, value_block, state.out[j]);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
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
        ushort lane) {
    for (short j = 0; j < 8; ++j) {
        simdgroup_store(state.out[j], stage + j * 8, SYNAPSE_HEAD_DIM);
    }

    if (lane % 4 == 0) {
        row_state[lane / 4] = state.running_sum;
        row_state[8 + lane / 4] = state.running_max;
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort idx = lane; idx < 8 * SYNAPSE_HEAD_DIM; idx += 32) {
        const ushort r = idx / SYNAPSE_HEAD_DIM;
        const ushort d = idx % SYNAPSE_HEAD_DIM;
        const attention_row row = attention_map(block, a.group, sg, r);
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

// Stages a simdgroup's 8x64 query rows (zero for invalid rows).
static inline void stage_query(constant attention_args & a, attention_block block, device const float * qkv,
                               threadgroup float * stage, ushort sg, ushort lane) {
    for (ushort idx = lane; idx < 8 * SYNAPSE_HEAD_DIM; idx += 32) {
        const attention_row row = attention_map(block, a.group, sg, idx / SYNAPSE_HEAD_DIM);
        stage[idx] = row.valid
            ? qkv[(ulong)row.token * a.qkv_stride + row.head * SYNAPSE_HEAD_DIM + idx % SYNAPSE_HEAD_DIM]
            : 0.0f;
    }
}

static inline attention_state attention_start() {
    attention_state state;
    for (short j = 0; j < 8; ++j) {
        state.out[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    state.running_max = -INFINITY;
    state.running_sum = 0.0f;
    return state;
}

// Prompt runs (mode 0). Grid: (run blocks, 1). Threadgroup: group * 32 threads, one simdgroup per query head.
// Every simdgroup reads K/V straight from the slot (the group's simdgroups hit the same cache lines) and owns a
// private threadgroup region: query staging first, then its score and rescale scratch. No threadgroup barrier is
// needed, and the small footprint lets several threadgroups share a core.
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
    const ulong key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, a.context);
    const ulong value_base = kv_value_base(a.layers, a.kv_heads, a.context) + key_base;

    threadgroup float * stage = region[sg];
    stage_query(a, block, qkv, stage, sg, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 query[8];
    for (short dk = 0; dk < 8; ++dk) {
        simdgroup_load(query[dk], stage + dk * 8, SYNAPSE_HEAD_DIM);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * scores = stage;
    threadgroup float * diagonal = stage + 8 * ATT_KEYS;
    attention_state state = attention_start();
    const attention_row mine = attention_map(block, a.group, sg, lane / 4);
    const uint my_position = mine.valid ? (uint)tokens[mine.token].position : 0;
    for (uint first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const uint count = min((uint)ATT_KEYS, key_end - first_key);
        const ulong offset = (ulong)first_key * SYNAPSE_HEAD_DIM;
        attend_tile<KV>(state, query, kv + key_base + offset, kv + value_base + offset, scores, diagonal, first_key,
                        count, mine.valid, my_position, a.scale, lane);
    }

    attention_finish(a, state, block, stage, row_state[sg], output, partial, tg.x, split, sg, lane);
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
    threadgroup float scratch[8 * ATT_KEYS];
    threadgroup float diagonal[64];
    threadgroup float row_state[16];

    const attention_block block = blocks[a.block_base + tg.x];
    const uint split = tg.y;
    const uint key_begin = min(block.key_end, split * a.split_keys);
    const uint key_end = min(block.key_end, key_begin + a.split_keys);
    device const KV * kv = (device const KV *)slots.address[block.slot];
    const ulong key_base = kv_offset(a.layer, block.kv_head, 0, a.kv_heads, a.context);
    const ulong value_base = kv_value_base(a.layers, a.kv_heads, a.context) + key_base;

    stage_query(a, block, qkv, stage, 0, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 query[8];
    for (short dk = 0; dk < 8; ++dk) {
        simdgroup_load(query[dk], stage + dk * 8, SYNAPSE_HEAD_DIM);
    }

    attention_state state = attention_start();
    const attention_row mine = attention_map(block, a.group, 0, lane / 4);
    const uint my_position = mine.valid ? (uint)tokens[mine.token].position : 0;
    for (uint first_key = key_begin; first_key < key_end; first_key += ATT_KEYS) {
        const uint count = min((uint)ATT_KEYS, key_end - first_key);
        const ulong offset = (ulong)first_key * SYNAPSE_HEAD_DIM;
        attend_tile<KV>(state, query, kv + key_base + offset, kv + value_base + offset, scratch, diagonal, first_key,
                    count, mine.valid, my_position, a.scale, lane);
    }

    attention_finish(a, state, block, stage, row_state, output, partial, tg.x, split, 0, lane);
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
