//! Attention dispatches: prompt-run blocks in watchdog-safe groups, then decode blocks with key splits.

use super::{Binding, Kernel, StepBackend, StepInputs};
use crate::gpu_kernels::decoder::HEAD_DIM;
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::params::AttentionArgs;

/// `1 / sqrt(HEAD_DIM)`, exact for a head dimension of 64 and equal to the reference operator.
const ATTENTION_SCALE: f32 = 0.125;
/// Query-key pairs of prompt-run attention per submission: about 40 ms on an M2 Pro, well under the display
/// watchdog, and enough blocks (about 47 at 88k keys) to keep every core busy for whole waves.
const ATTENTION_GROUP_PAIRS: u64 = 1 << 25;

/// Encodes attention for layer `layer` of the step.
pub(super) fn attention<S: StepBackend>(
    backend: &mut S,
    inputs: &StepInputs<'_, S::Buffer>,
    layer: u32,
) -> Result<(), GpuError> {
    let shape = inputs.plan.shape;
    let base = AttentionArgs {
        heads: shape.heads,
        kv_heads: shape.kv_heads,
        group: shape.group(),
        qkv_stride: shape.qkv_width(),
        out_stride: shape.hidden,
        layer,
        layers: shape.layer_count,
        context: shape.context,
        split_keys: shape.context,
        splits: 1,
        block_base: 0,
        pad: 0,
        scale: ATTENTION_SCALE,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    prompt_runs(backend, inputs, &base)?;
    decode_blocks(backend, inputs, &base)
}

const fn bindings<'a, B>(inputs: &StepInputs<'a, B>) -> [Binding<'a, B>; 6] {
    let a = inputs.activations;
    [
        Binding::whole(1, &a.qkv),
        Binding::whole(2, &a.attention),
        Binding::whole(3, &a.tokens),
        Binding::Table {
            index: 4,
            table: inputs.table,
        },
        Binding::whole(5, &a.blocks),
        Binding::whole(6, &a.partial),
    ]
}

/// Prompt-run blocks go out in groups of at most `ATTENTION_GROUP_PAIRS` query-key pairs, each its own
/// submission, so no single submission holds the GPU long enough to trip the display watchdog.
fn prompt_runs<S: StepBackend>(
    backend: &mut S,
    inputs: &StepInputs<'_, S::Buffer>,
    base: &AttentionArgs,
) -> Result<(), GpuError> {
    let plan = inputs.attention;
    let shape = inputs.plan.shape;
    let mut first = 0;
    while first < plan.run_blocks {
        let mut last = first;
        let mut pairs = 0_u64;
        while last < plan.run_blocks {
            let block = plan.blocks[last as usize];
            let cost = u64::from(block.token_count) * u64::from(block.key_end);
            if last > first && pairs + cost > ATTENTION_GROUP_PAIRS {
                break;
            }

            pairs += cost;
            last += 1;
        }

        if first > 0 {
            backend.flush()?;
        }

        let group = AttentionArgs {
            block_base: first,
            ..*base
        };
        backend.launch(
            Kernel::Attention(shape.kv_precision),
            &group,
            &bindings(inputs),
            [(last - first) as usize, 1, 1],
            [32 * shape.group() as usize, 1, 1],
        )?;
        first = last;
    }

    Ok(())
}

/// Decode blocks run one simdgroup each over key splits; more than one split is merged by the reduce kernel.
fn decode_blocks<S: StepBackend>(
    backend: &mut S,
    inputs: &StepInputs<'_, S::Buffer>,
    base: &AttentionArgs,
) -> Result<(), GpuError> {
    let plan = inputs.attention;
    if plan.single_blocks == 0 {
        return Ok(());
    }

    let a = inputs.activations;
    let singles = AttentionArgs {
        split_keys: plan.split_keys,
        splits: plan.splits,
        block_base: plan.run_blocks,
        ..*base
    };
    backend.launch(
        Kernel::AttentionDecode(inputs.plan.shape.kv_precision),
        &singles,
        &bindings(inputs),
        [plan.single_blocks as usize, plan.splits as usize, 1],
        [32, 1, 1],
    )?;
    if plan.splits == 1 {
        return Ok(());
    }

    backend.launch(
        Kernel::AttentionReduce,
        &singles,
        &[
            Binding::whole(1, &a.blocks),
            Binding::whole(2, &a.partial),
            Binding::whole(3, &a.attention),
        ],
        [1, 64, plan.single_blocks as usize],
        [HEAD_DIM as usize, 1, 1],
    )
}
