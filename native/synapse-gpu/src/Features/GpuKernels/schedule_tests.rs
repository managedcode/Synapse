use super::{ATTENTION_KEYS, PARTIAL_BLOCKS, plan_attention};
use crate::gpu_kernels::decoder::BatchToken;

fn token(slot: i32, position: i32) -> BatchToken {
    BatchToken {
        slot,
        token: 1,
        position,
        logits_row: -1,
    }
}

#[test]
fn prompt_runs_split_into_eight_token_blocks_per_kv_head() {
    let tokens: Vec<_> = (0..20).map(|position| token(0, position)).collect();

    let plan = plan_attention(&tokens, 2);

    assert_eq!(plan.run_blocks, 6);
    assert_eq!(plan.single_blocks, 0);
    assert_eq!(plan.splits, 1);
    let counts: Vec<_> = plan
        .blocks
        .iter()
        .map(|block| (block.first_token, block.token_count, block.kv_head))
        .collect();
    assert_eq!(
        counts,
        [
            (0, 8, 0),
            (0, 8, 1),
            (8, 8, 0),
            (8, 8, 1),
            (16, 4, 0),
            (16, 4, 1)
        ]
    );
    assert!(plan.blocks.iter().all(|block| block.mode == 0));
    assert_eq!(plan.blocks[5].key_end, 20);
}

#[test]
fn decode_tokens_become_single_blocks_after_prompt_runs() {
    let tokens = [
        token(1, 40),
        token(2, 7),
        token(3, 0),
        token(3, 1),
        token(3, 2),
    ];

    let plan = plan_attention(&tokens, 2);

    assert_eq!(plan.run_blocks, 2);
    assert_eq!(plan.single_blocks, 4);
    assert_eq!(plan.blocks[0].first_token, 2);
    assert_eq!(plan.blocks[0].key_end, 3);
    let singles: Vec<_> = plan.blocks[2..]
        .iter()
        .map(|block| (block.first_token, block.slot, block.key_end, block.mode))
        .collect();
    assert_eq!(
        singles,
        [(0, 1, 41, 1), (0, 1, 41, 1), (1, 2, 8, 1), (1, 2, 8, 1)]
    );
}

#[test]
fn non_consecutive_positions_break_a_run() {
    let tokens = [token(0, 0), token(0, 1), token(0, 5), token(0, 6)];

    let plan = plan_attention(&tokens, 1);

    assert_eq!(plan.run_blocks, 2);
    assert_eq!(plan.blocks[1].first_token, 2);
    assert_eq!(plan.blocks[1].key_end, 7);
}

#[test]
fn long_decode_contexts_split_within_the_partial_buffer() {
    let one = plan_attention(&[token(0, 32_767)], 2);
    let many: Vec<_> = (0..64).map(|slot| token(slot, 131_071)).collect();
    let crowded = plan_attention(&many, 2);
    let short = plan_attention(&[token(0, 3)], 2);

    assert!(one.splits > 1);
    assert_eq!(one.split_keys % ATTENTION_KEYS, 0);
    assert!(one.split_keys * one.splits >= 32_768);
    assert!(one.single_blocks * one.splits <= PARTIAL_BLOCKS);
    assert!(crowded.single_blocks * crowded.splits <= PARTIAL_BLOCKS);
    assert!(crowded.split_keys * crowded.splits >= 131_072);
    assert_eq!(short.splits, 1);
    assert!(short.split_keys >= 4);
}
