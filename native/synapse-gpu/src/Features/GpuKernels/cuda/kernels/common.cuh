// Synapse CUDA kernels for dense Q8_0 decoders (ADR-012). Compiled at load with NVRTC, so the build needs no
// CUDA toolkit. Every struct mirrors a `#[repr(C)]` type in `params.rs` and the Metal `common.metal`; the
// kernels implement the same algorithms, launch geometry, and parameter order as the Metal kernels.

typedef unsigned long long u64;
typedef unsigned int u32;

#define SYNAPSE_MAX_SLOTS 65
#define SYNAPSE_HEAD_DIM 64
#define SYNAPSE_NO_BIAS 0xFFFFFFFFFFFFFFFFull
#define SYNAPSE_FULL_MASK 0xFFFFFFFFu

struct block_q8_0 {
    unsigned short d;
    signed char qs[32];
};

struct batch_token {
    int slot;
    int token;
    int position;
    int logits_row;
};

struct slot_table {
    float * address[SYNAPSE_MAX_SLOTS];
};

struct embed_args {
    u64 table;
    u32 columns;
    u32 row_bytes;
    u32 tokens;
    u32 out_stride;
};

struct norm_args {
    u64 weight;
    u32 columns;
    u32 rows;
    u32 in_stride;
    u32 out_stride;
    float epsilon;
    u32 gather_logits;
    u32 token_count;
    u32 pad;
};

struct matmul_segment {
    u64 weight;
    u64 bias;
    u32 rows;
    u32 out_column;
};

struct matmul_args {
    matmul_segment segment[3];
    u32 segments;
    u32 total_rows;
    u32 columns;
    u32 tokens;
    u32 in_stride;
    u32 out_stride;
    u32 accumulate;
    u32 pad;
};

struct rope_args {
    u32 tokens;
    u32 heads;
    u32 kv_heads;
    u32 half_dim;
    u32 qkv_stride;
    u32 key_column;
    u32 value_column;
    u32 layer;
    u32 layers;
    u32 context;
};

struct attention_block {
    u32 first_token;
    u32 token_count;
    u32 kv_head;
    u32 mode;
    u32 slot;
    u32 key_end;
    u32 pad0;
    u32 pad1;
};

struct attention_args {
    u32 heads;
    u32 kv_heads;
    u32 group;
    u32 qkv_stride;
    u32 out_stride;
    u32 layer;
    u32 layers;
    u32 context;
    u32 split_keys;
    u32 splits;
    u32 block_base;
    u32 pad;
    float scale;
    u32 pad0;
    u32 pad1;
    u32 pad2;
};

struct swiglu_args {
    u32 columns;
    u32 tokens;
    u32 in_stride;
    u32 out_stride;
};

__device__ __forceinline__ float half_to_float(unsigned short bits) {
    float value;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(value) : "h"(bits));
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(SYNAPSE_FULL_MASK, value, offset);
    }

    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(SYNAPSE_FULL_MASK, value, offset));
    }

    return value;
}

__device__ __forceinline__ u64 kv_offset(u32 layer, u32 kv_head, u32 position, u32 kv_heads, u32 context) {
    return ((((u64)layer * kv_heads) + kv_head) * context + position) * SYNAPSE_HEAD_DIM;
}

__device__ __forceinline__ u64 kv_value_base(u32 layers, u32 kv_heads, u32 context) {
    return (u64)layers * kv_heads * context * SYNAPSE_HEAD_DIM;
}

// Row inside one weight segment: returns the segment index and rewrites `row` to the local row.
__device__ __forceinline__ u32 locate_segment(const matmul_args & a, u32 & row) {
    u32 s = 0;
    while (s + 1 < a.segments && row >= a.segment[s].rows) {
        row -= a.segment[s].rows;
        ++s;
    }

    return s;
}
