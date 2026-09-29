//! Groups one step's tokens into attention work units.
//!
//! Consecutive tokens of one slot with consecutive positions form a prompt run (mode 0: one simdgroup
//! per query head, rows are tokens); a lone token is a decode unit (mode 1: one simdgroup, rows are the group's query heads) whose keys may be split across
//! threadgroups for long contexts.

use super::params::AttentionBlock;
use crate::gpu_kernels::decoder::BatchToken;

/// Prompt-run tokens per block on backends whose attention holds one eight-row half per simdgroup (CUDA).
pub const RUN_TOKENS: usize = 8;
/// Prompt-run tokens per block on Metal: `ATT_RUN_HALVES` eight-row halves per simdgroup. Two halves share each K/V
/// load but spilled registers on an M2 Pro, so Metal uses one.
pub const METAL_RUN_TOKENS: usize = 8;
/// Keys per attention tile.
pub const ATTENTION_KEYS: u32 = 32;
/// Block-split pairs the partial buffer holds.
pub const PARTIAL_BLOCKS: u32 = 512;
const TARGET_GROUPS: u32 = 256;
const MIN_SPLIT_KEYS: u32 = 256;

/// Attention work for one step: prompt-run blocks first, then decode blocks with their split plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPlan {
    pub blocks: Vec<AttentionBlock>,
    pub run_blocks: u32,
    pub single_blocks: u32,
    pub split_keys: u32,
    pub splits: u32,
}

/// Builds the attention plan for validated step tokens; prompt runs are cut into blocks of `run_tokens`.
#[must_use]
pub fn plan_attention(tokens: &[BatchToken], kv_heads: u32, run_tokens: usize) -> AttentionPlan {
    let mut runs = Vec::new();
    let mut singles = Vec::new();
    let mut start = 0;
    while start < tokens.len() {
        let mut end = start + 1;
        while end < tokens.len()
            && tokens[end].slot == tokens[start].slot
            && tokens[end].position == tokens[end - 1].position + 1
        {
            end += 1;
        }

        if end - start == 1 {
            singles.push(start);
        } else {
            let mut first = start;
            while first < end {
                let count = run_tokens.min(end - first);
                runs.push((first, count));
                first += count;
            }
        }

        start = end;
    }

    let mut blocks = Vec::with_capacity((runs.len() + singles.len()) * kv_heads as usize);
    for &(first, count) in &runs {
        let last = tokens[first + count - 1];
        push_blocks(&mut blocks, kv_heads, first, count, 0, last);
    }

    let run_blocks = to_u32(blocks.len());
    for &index in &singles {
        push_blocks(&mut blocks, kv_heads, index, 1, 1, tokens[index]);
    }

    let single_blocks = to_u32(blocks.len()) - run_blocks;
    let longest = blocks[run_blocks as usize..]
        .iter()
        .map(|block| block.key_end)
        .max()
        .unwrap_or(0);
    let (split_keys, splits) = split_plan(single_blocks, longest);
    AttentionPlan {
        blocks,
        run_blocks,
        single_blocks,
        split_keys,
        splits,
    }
}

fn push_blocks(
    blocks: &mut Vec<AttentionBlock>,
    kv_heads: u32,
    first: usize,
    count: usize,
    mode: u32,
    last: BatchToken,
) {
    for kv_head in 0..kv_heads {
        blocks.push(AttentionBlock {
            first_token: to_u32(first),
            token_count: to_u32(count),
            kv_head,
            mode,
            slot: last.slot.unsigned_abs(),
            key_end: last.position.unsigned_abs() + 1,
            pad0: 0,
            pad1: 0,
        });
    }
}

/// Splits the longest decode context so that about `TARGET_GROUPS` threadgroups run, each over at least
/// `MIN_SPLIT_KEYS` keys, within the partial buffer.
fn split_plan(blocks: u32, longest: u32) -> (u32, u32) {
    if blocks == 0 || longest == 0 {
        return (ATTENTION_KEYS, 1);
    }

    let wanted = TARGET_GROUPS.div_ceil(blocks);
    let by_keys = longest.div_ceil(MIN_SPLIT_KEYS);
    let by_buffer = (PARTIAL_BLOCKS / blocks).max(1);
    let splits = wanted.min(by_keys).min(by_buffer).max(1);
    let split_keys = longest.div_ceil(splits).next_multiple_of(ATTENTION_KEYS);
    (split_keys, longest.div_ceil(split_keys))
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
