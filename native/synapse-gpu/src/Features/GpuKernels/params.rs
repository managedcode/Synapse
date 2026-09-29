//! `#[repr(C)]` mirrors of the argument structs in `shaders/common.metal`. Field order and sizes must match;
//! the size assertions below catch accidental drift.

pub const NO_BIAS: u64 = u64::MAX;
pub const MAX_SLOTS: usize = 65;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SlotTable {
    pub address: [u64; MAX_SLOTS],
    /// Allocated positions of each slot: the position stride of its K and V regions (ADR-017).
    pub capacity: [u32; MAX_SLOTS],
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedArgs {
    pub table: u64,
    pub columns: u32,
    pub row_bytes: u32,
    pub tokens: u32,
    pub out_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NormArgs {
    pub weight: u64,
    pub columns: u32,
    pub rows: u32,
    pub in_stride: u32,
    pub out_stride: u32,
    pub epsilon: f32,
    pub gather_logits: u32,
    pub token_count: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MatmulSegment {
    pub weight: u64,
    pub bias: u64,
    pub rows: u32,
    pub out_column: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatmulArgs {
    pub segment: [MatmulSegment; 3],
    pub segments: u32,
    pub total_rows: u32,
    pub columns: u32,
    pub tokens: u32,
    pub in_stride: u32,
    pub out_stride: u32,
    pub accumulate: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopeArgs {
    pub tokens: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub half_dim: u32,
    pub qkv_stride: u32,
    pub key_column: u32,
    pub value_column: u32,
    pub layer: u32,
    pub layers: u32,
    pub context: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionBlock {
    pub first_token: u32,
    pub token_count: u32,
    pub kv_head: u32,
    pub mode: u32,
    pub slot: u32,
    pub key_end: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttentionArgs {
    pub heads: u32,
    pub kv_heads: u32,
    pub group: u32,
    pub qkv_stride: u32,
    pub out_stride: u32,
    pub layer: u32,
    pub layers: u32,
    pub context: u32,
    pub split_keys: u32,
    pub splits: u32,
    pub block_base: u32,
    pub pad: u32,
    pub scale: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SwigluArgs {
    pub columns: u32,
    pub tokens: u32,
    pub in_stride: u32,
    pub out_stride: u32,
}

const _: () = {
    assert!(size_of::<SlotTable>() == 784);
    assert!(size_of::<EmbedArgs>() == 24);
    assert!(size_of::<NormArgs>() == 40);
    assert!(size_of::<MatmulSegment>() == 24);
    assert!(size_of::<MatmulArgs>() == 104);
    assert!(size_of::<RopeArgs>() == 40);
    assert!(size_of::<AttentionBlock>() == 32);
    assert!(size_of::<AttentionArgs>() == 64);
    assert!(size_of::<SwigluArgs>() == 16);
};
