// Embedding lookup, RMS normalization, RoPE with KV append, and SwiGLU.

// One thread per (column, token): hidden[t][c] = d * q of the token's Q8_0 embedding row.
kernel void synapse_embed_q8_0(
        constant embed_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const batch_token * tokens [[buffer(2)]],
        device float * hidden [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.columns || gid.y >= a.tokens) {
        return;
    }

    const ulong row = (ulong)tokens[gid.y].token;
    device const block_q8_0 * blocks = (device const block_q8_0 *)(weights + a.table + row * a.row_bytes);
    const block_q8_0 block = blocks[gid.x / 32];
    hidden[(ulong)gid.y * a.out_stride + gid.x] = (float)block.d * (float)block.qs[gid.x % 32];
}

// One threadgroup per row: y = x / sqrt(mean(x^2) + eps) * w. In gather mode, output row r normalizes
// input row gather[r] (the batch token that owns logits row r).
kernel void synapse_rms_norm(
        constant norm_args & a [[buffer(0)]],
        device const char * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        device const uint * gather [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    if (row >= a.rows) {
        return;
    }

    const ulong source = a.gather_logits != 0 ? (ulong)gather[row] : (ulong)row;
    device const float * x = input + source * a.in_stride;
    device const float * w = (device const float *)(weights + a.weight);
    float sum = 0.0f;
    for (uint i = tid; i < a.columns; i += threads) {
        sum += x[i] * x[i];
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        partial[sg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint groups = (threads + 31) / 32;
        float total = lane < groups ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            partial[0] = total;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(partial[0] / (float)a.columns + a.epsilon);
    device float * y = output + (ulong)row * a.out_stride;
    for (uint i = tid; i < a.columns; i += threads) {
        y[i] = x[i] * scale * w[i];
    }
}

// Grid (half_dim, heads + 2 * kv_heads, tokens). Query heads rotate in place; key heads rotate and append to
// the token's KV slot; value heads append unchanged. Cosines and sines come from the C# table (ADR-013). KV is
// the slot element type: float, or half for the explicit FP16 KV profile.
template <typename KV>
kernel void synapse_rope_kv(
        constant rope_args & a [[buffer(0)]],
        device float * qkv [[buffer(1)]],
        device const float * cosines [[buffer(2)]],
        device const float * sines [[buffer(3)]],
        device const batch_token * tokens [[buffer(4)]],
        constant slot_table & slots [[buffer(5)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    const uint head = gid.y;
    if (i >= a.half_dim || head >= a.heads + 2 * a.kv_heads || gid.z >= a.tokens) {
        return;
    }

    const batch_token token = tokens[gid.z];
    device float * row = qkv + (ulong)gid.z * a.qkv_stride;
    device KV * kv = (device KV *)slots.address[token.slot];
    const uint position = (uint)token.position;
    if (head >= a.heads + a.kv_heads) {
        const uint kv_head = head - a.heads - a.kv_heads;
        device const float * value = row + a.value_column + kv_head * SYNAPSE_HEAD_DIM;
        const ulong base = kv_value_base(a.layers, a.kv_heads, a.context) +
            kv_offset(a.layer, kv_head, position, a.kv_heads, a.context);
        kv[base + i] = (KV)value[i];
        kv[base + i + a.half_dim] = (KV)value[i + a.half_dim];
        return;
    }

    const float c = cosines[(ulong)position * a.half_dim + i];
    const float s = sines[(ulong)position * a.half_dim + i];
    const bool is_key = head >= a.heads;
    device float * values = is_key
        ? row + a.key_column + (head - a.heads) * SYNAPSE_HEAD_DIM
        : row + head * SYNAPSE_HEAD_DIM;
    const float first = values[i];
    const float second = values[i + a.half_dim];
    const float rotated_first = first * c - second * s;
    const float rotated_second = first * s + second * c;
    values[i] = rotated_first;
    values[i + a.half_dim] = rotated_second;
    if (is_key) {
        const ulong base = kv_offset(a.layer, head - a.heads, position, a.kv_heads, a.context);
        kv[base + i] = (KV)rotated_first;
        kv[base + i + a.half_dim] = (KV)rotated_second;
    }
}

template [[host_name("synapse_rope_kv")]] kernel void synapse_rope_kv<float>(
    constant rope_args &, device float *, device const float *, device const float *, device const batch_token *,
    constant slot_table &, uint3);
template [[host_name("synapse_rope_kv_f16")]] kernel void synapse_rope_kv<half>(
    constant rope_args &, device float *, device const float *, device const float *, device const batch_token *,
    constant slot_table &, uint3);

// h[t][c] = silu(gate[t][c]) * up[t][c], where gate and up are the two halves of one input row.
kernel void synapse_swiglu(
        constant swiglu_args & a [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.columns || gid.y >= a.tokens) {
        return;
    }

    device const float * row = input + (ulong)gid.y * a.in_stride;
    const float gate = row[gid.x];
    output[(ulong)gid.y * a.out_stride + gid.x] = gate / (1.0f + exp(-gate)) * row[a.columns + gid.x];
}
