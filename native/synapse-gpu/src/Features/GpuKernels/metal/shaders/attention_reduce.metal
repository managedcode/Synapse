// Merges the key splits of long decode contexts (ADR-012): the standard max/sum rescaling over each split's
// unnormalized output. Uses attention_map and ATT_ROW_FLOATS from attention.metal, which precedes this source.

// Grid (head dims, 64 rows, blocks): merges the splits of one decode dispatch into the attention output.
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
