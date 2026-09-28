// Embedding lookup, RMS normalization, RoPE with KV append, and SwiGLU (same semantics as elementwise.metal).

extern "C" __global__ void synapse_embed_q8_0(
        embed_args a, const char * weights, const batch_token * tokens, float * hidden) {
    const u32 column = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 token = blockIdx.y;
    if (column >= a.columns || token >= a.tokens) {
        return;
    }

    const block_q8_0 * blocks =
        (const block_q8_0 *)(weights + a.table + (u64)tokens[token].token * a.row_bytes);
    const block_q8_0 & block = blocks[column / 32];
    hidden[(u64)token * a.out_stride + column] = half_to_float(block.d) * (float)block.qs[column % 32];
}

extern "C" __global__ void synapse_rms_norm(
        norm_args a, const char * weights, const float * input, float * output, const u32 * gather) {
    __shared__ float partial[32];
    const u32 row = blockIdx.x;
    if (row >= a.rows) {
        return;
    }

    const u64 source = a.gather_logits != 0 ? (u64)gather[row] : (u64)row;
    const float * x = input + source * a.in_stride;
    const float * w = (const float *)(weights + a.weight);
    float sum = 0.0f;
    for (u32 i = threadIdx.x; i < a.columns; i += blockDim.x) {
        sum += x[i] * x[i];
    }

    const u32 lane = threadIdx.x & 31;
    const u32 warp = threadIdx.x >> 5;
    sum = warp_sum(sum);
    if (lane == 0) {
        partial[warp] = sum;
    }

    __syncthreads();
    if (warp == 0) {
        const u32 groups = (blockDim.x + 31) / 32;
        float total = lane < groups ? partial[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }

    __syncthreads();
    const float scale = 1.0f / sqrtf(partial[0] / (float)a.columns + a.epsilon);
    float * y = output + (u64)row * a.out_stride;
    for (u32 i = threadIdx.x; i < a.columns; i += blockDim.x) {
        y[i] = x[i] * scale * w[i];
    }
}

// Grid (1, heads + 2 * kv_heads, tokens) of half_dim threads.
extern "C" __global__ void synapse_rope_kv(
        rope_args a, float * qkv, const float * cosines, const float * sines, const batch_token * tokens,
        slot_table slots) {
    const u32 i = threadIdx.x;
    const u32 head = blockIdx.y;
    const u32 index = blockIdx.z;
    if (i >= a.half_dim || head >= a.heads + 2 * a.kv_heads || index >= a.tokens) {
        return;
    }

    const batch_token token = tokens[index];
    float * row = qkv + (u64)index * a.qkv_stride;
    float * kv = slots.address[token.slot];
    const u32 position = (u32)token.position;
    if (head >= a.heads + a.kv_heads) {
        const u32 kv_head = head - a.heads - a.kv_heads;
        const float * value = row + a.value_column + kv_head * SYNAPSE_HEAD_DIM;
        const u64 base = kv_value_base(a.layers, a.kv_heads, a.context) +
            kv_offset(a.layer, kv_head, position, a.kv_heads, a.context);
        kv[base + i] = value[i];
        kv[base + i + a.half_dim] = value[i + a.half_dim];
        return;
    }

    const float c = cosines[(u64)position * a.half_dim + i];
    const float s = sines[(u64)position * a.half_dim + i];
    const bool is_key = head >= a.heads;
    float * values = is_key ? row + a.key_column + (head - a.heads) * SYNAPSE_HEAD_DIM : row + head * SYNAPSE_HEAD_DIM;
    const float first = values[i];
    const float second = values[i + a.half_dim];
    const float rotated_first = first * c - second * s;
    const float rotated_second = first * s + second * c;
    values[i] = rotated_first;
    values[i + a.half_dim] = rotated_second;
    if (is_key) {
        const u64 base = kv_offset(a.layer, head - a.heads, position, a.kv_heads, a.context);
        kv[base + i] = rotated_first;
        kv[base + i + a.half_dim] = rotated_second;
    }
}

extern "C" __global__ void synapse_swiglu(swiglu_args a, const float * input, float * output) {
    const u32 column = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 token = blockIdx.y;
    if (column >= a.columns || token >= a.tokens) {
        return;
    }

    const float * row = input + (u64)token * a.in_stride;
    const float gate = row[column];
    output[(u64)token * a.out_stride + column] = gate / (1.0f + expf(-gate)) * row[a.columns + column];
}
