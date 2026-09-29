// Synapse Metal kernels for dense Q8_0 decoders (ADR-012). Every struct below mirrors a `#[repr(C)]`
// type in `metal/params.rs`; field order and sizes must stay identical.
#include <metal_stdlib>
using namespace metal;

#define SYNAPSE_MAX_SLOTS 65
// The host compiles the library per model and defines SYNAPSE_HEAD_DIM (64 or 128) before this source.
#ifndef SYNAPSE_HEAD_DIM
#error "SYNAPSE_HEAD_DIM must be defined by the host"
#endif
#define SYNAPSE_NO_BIAS 0xFFFFFFFFFFFFFFFFul

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

struct batch_token {
    int slot;
    int token;
    int position;
    int logits_row;
};

// GPU addresses (`MTLBuffer.gpuAddress`) of each KV slot buffer, null for an unallocated slot, and each slot's
// allocated positions, which are also the position stride of its K and V regions (ADR-017).
struct slot_table {
    device float * address[SYNAPSE_MAX_SLOTS];
    uint capacity[SYNAPSE_MAX_SLOTS];
    uint pad;
};

struct embed_args {
    ulong table;
    uint columns;
    uint row_bytes;
    uint tokens;
    uint out_stride;
};

struct norm_args {
    ulong weight;
    uint columns;
    uint rows;
    uint in_stride;
    uint out_stride;
    float epsilon;
    // 1: row r reads input row `tokens[r].logits_row` order (logits gather); 0: row r reads row r.
    uint gather_logits;
    uint token_count;
    uint pad;
};

struct matmul_segment {
    ulong weight;
    ulong bias;
    uint rows;
    uint out_column;
};

struct matmul_args {
    matmul_segment segment[3];
    uint segments;
    uint total_rows;
    uint columns;
    uint tokens;
    uint in_stride;
    uint out_stride;
    uint accumulate;
    uint pad;
};

struct rope_args {
    uint tokens;
    uint heads;
    uint kv_heads;
    uint half_dim;
    uint qkv_stride;
    uint key_column;
    uint value_column;
    uint layer;
    uint layers;
    uint context;
};

// A work unit of the attention kernel: `token_count` consecutive batch tokens of one slot and one KV head.
// Mode 0 maps simdgroup s to query head `kv_head * group + s` and its eight rows to tokens; mode 1 maps
// simdgroup s to one token and its rows to the group's query heads.
struct attention_block {
    uint first_token;
    uint token_count;
    uint kv_head;
    uint mode;
    uint slot;
    uint key_end;
    uint pad0;
    uint pad1;
};

struct attention_args {
    uint heads;
    uint kv_heads;
    uint group;
    uint qkv_stride;
    uint out_stride;
    uint layer;
    uint layers;
    uint context;
    // Keys handled by one split and the number of splits; one split writes the final output directly.
    uint split_keys;
    uint splits;
    // Index of this dispatch's first block in the shared block and partial buffers.
    uint block_base;
    uint pad;
    float scale;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct swiglu_args {
    uint columns;
    uint tokens;
    uint in_stride;
    uint out_stride;
};

// Element offset of (layer, kv_head, position, 0) in a slot's FP32 key region; values follow all keys.
static inline ulong kv_offset(uint layer, uint kv_head, uint position, uint kv_heads, uint context) {
    return ((((ulong)layer * kv_heads) + kv_head) * context + position) * SYNAPSE_HEAD_DIM;
}

static inline ulong kv_value_base(uint layers, uint kv_heads, uint context) {
    return (ulong)layers * kv_heads * context * SYNAPSE_HEAD_DIM;
}

// Row `row` of the concatenated segments: returns the segment index and rewrites `row` to its local row.
static inline uint locate_segment(constant matmul_args & a, thread uint & row) {
    uint s = 0;
    while (s + 1 < a.segments && row >= a.segment[s].rows) {
        row -= a.segment[s].rows;
        ++s;
    }

    return s;
}
