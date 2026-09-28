//! Projections: decode tokens on the batch-invariant matrix-vector kernel, prompt runs on the tiled GEMM.

use super::{Binding, Kernel, StepBackend};
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::params::{MatmulArgs, MatmulSegment};

const MATVEC_ROWS: u32 = 2;
const MATVEC_THREADS: usize = 128;
const GEMM_ROWS: u32 = 64;
const GEMM_TOKENS: u32 = 32;
const GEMM_THREADS: usize = 128;
/// Prompt runs of at most this many tokens stay on the matrix-vector kernel.
const GEMM_MINIMUM_TOKENS: u32 = 8;

/// One projection: up to three weight segments over `span = (tokens, first prompt-run token)`.
pub(super) struct Projection<'a> {
    pub segments: &'a [MatmulSegment],
    pub columns: u32,
    pub span: (u32, u32),
    pub strides: [u32; 2],
    pub accumulate: bool,
}

impl Projection<'_> {
    fn args(&self, tokens: u32) -> MatmulArgs {
        let mut segment = [MatmulSegment::default(); 3];
        segment[..self.segments.len()].copy_from_slice(self.segments);
        MatmulArgs {
            segment,
            segments: u32::try_from(self.segments.len()).unwrap_or(3),
            total_rows: self.total_rows(),
            columns: self.columns,
            tokens,
            in_stride: self.strides[0],
            out_stride: self.strides[1],
            accumulate: u32::from(self.accumulate),
            pad: 0,
        }
    }

    fn total_rows(&self) -> u32 {
        self.segments.iter().map(|segment| segment.rows).sum()
    }
}

/// Leading single tokens (decode) use the matrix-vector kernel; the prompt-run tokens from the span's second
/// element use the tiled GEMM when there are more than eight of them.
pub(super) fn project<S: StepBackend>(
    backend: &mut S,
    weights: &S::Buffer,
    projection: &Projection<'_>,
    buffers: [&S::Buffer; 2],
) -> Result<(), GpuError> {
    let (tokens, gemm_from) = projection.span;
    let split = if tokens - gemm_from.min(tokens) > GEMM_MINIMUM_TOKENS {
        gemm_from
    } else {
        tokens
    };
    if split > 0 {
        matrix_vector(backend, weights, projection, buffers, split)?;
    }

    if split < tokens {
        gemm(backend, weights, projection, buffers, split)?;
    }

    Ok(())
}

/// Tokens `0..count` on the smallest matrix-vector instance that holds them (at most eight per group).
fn matrix_vector<S: StepBackend>(
    backend: &mut S,
    weights: &S::Buffer,
    projection: &Projection<'_>,
    [input, output]: [&S::Buffer; 2],
    count: u32,
) -> Result<(), GpuError> {
    let per_group = match count {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        _ => 8,
    };
    backend.launch(
        Kernel::Matvec(per_group),
        &projection.args(count),
        &[
            Binding::whole(1, weights),
            Binding::whole(2, input),
            Binding::whole(3, output),
        ],
        [
            projection.total_rows().div_ceil(MATVEC_ROWS) as usize,
            count.div_ceil(per_group) as usize,
            1,
        ],
        [MATVEC_THREADS, 1, 1],
    )
}

/// Tokens `first..tokens` on the tiled GEMM, bound at their row offsets.
fn gemm<S: StepBackend>(
    backend: &mut S,
    weights: &S::Buffer,
    projection: &Projection<'_>,
    [input, output]: [&S::Buffer; 2],
    first: u32,
) -> Result<(), GpuError> {
    let rest = projection.span.0 - first;
    let [in_stride, out_stride] = projection.strides.map(|stride| stride as usize * 4);
    backend.launch(
        Kernel::Gemm,
        &projection.args(rest),
        &[
            Binding::whole(1, weights),
            Binding::Buffer {
                index: 2,
                buffer: input,
                offset: first as usize * in_stride,
            },
            Binding::Buffer {
                index: 3,
                buffer: output,
                offset: first as usize * out_stride,
            },
        ],
        [
            projection.total_rows().div_ceil(GEMM_ROWS) as usize,
            rest.div_ceil(GEMM_TOKENS) as usize,
            1,
        ],
        [GEMM_THREADS, 1, 1],
    )
}
