//! The per-layer kernel sequence: norms, projections, `RoPE` with KV append, attention, and `SwiGLU`.

use super::attention::attention;
use super::projection::{Projection, project};
use super::{Binding, Kernel, StepBackend, StepInputs};
use crate::gpu_kernels::decoder::{DecoderLayerOffsets, HEAD_DIM, Q8_0_BLOCK_VALUES};
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::params::{
    EmbedArgs, MatmulSegment, NO_BIAS, NormArgs, RopeArgs, SwigluArgs,
};

const NORM_THREADS: usize = 256;
const ELEMENT_THREADS: usize = 256;
const BLOCK_BYTES: u32 = 34;

/// One step in progress; `span` is (tokens, first prompt-run token).
pub(super) struct Step<'s, 'a, S: StepBackend> {
    pub backend: &'s mut S,
    pub inputs: &'s StepInputs<'a, S::Buffer>,
    pub span: (u32, u32),
}

impl<S: StepBackend> Step<'_, '_, S> {
    const fn weight(&self, offset: u64) -> u64 {
        self.inputs.weights_offset + offset
    }

    pub(super) fn embed(&mut self) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        let args = EmbedArgs {
            table: self.weight(self.inputs.plan.token_embedding),
            columns: shape.hidden,
            row_bytes: (shape.hidden / Q8_0_BLOCK_VALUES) * BLOCK_BYTES,
            tokens: self.span.0,
            out_stride: shape.hidden,
        };
        self.backend.launch(
            Kernel::Embed,
            &args,
            &[
                Binding::whole(1, self.inputs.weights),
                Binding::whole(2, &a.tokens),
                Binding::whole(3, &a.hidden),
            ],
            [
                (shape.hidden as usize).div_ceil(ELEMENT_THREADS),
                self.span.0 as usize,
                1,
            ],
            [ELEMENT_THREADS, 1, 1],
        )
    }

    pub(super) fn attention_half(
        &mut self,
        layer: u32,
        offsets: &DecoderLayerOffsets,
    ) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        let kv = shape.kv_width();
        self.norm(offsets.attention_norm)?;
        let qkv = [
            self.segment(offsets.query, offsets.query_bias, shape.hidden, 0),
            self.segment(offsets.key, offsets.key_bias, kv, shape.hidden),
            self.segment(offsets.value, offsets.value_bias, kv, shape.hidden + kv),
        ];
        self.project(
            &qkv,
            shape.hidden,
            [&a.normalized, &a.qkv],
            [shape.hidden, shape.qkv_width()],
            false,
        )?;
        self.rope(layer)?;
        attention(self.backend, self.inputs, layer)?;
        let output = [self.segment(offsets.attention_output, NO_BIAS, shape.hidden, 0)];
        self.project(
            &output,
            shape.hidden,
            [&a.attention, &a.hidden],
            [shape.hidden, shape.hidden],
            true,
        )
    }

    fn rope(&mut self, layer: u32) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        let kv = shape.kv_width();
        let count = self.span.0;
        let rope = RopeArgs {
            tokens: count,
            heads: shape.heads,
            kv_heads: shape.kv_heads,
            half_dim: HEAD_DIM / 2,
            qkv_stride: shape.qkv_width(),
            key_column: shape.hidden,
            value_column: shape.hidden + kv,
            layer,
            layers: shape.layer_count,
            context: shape.context,
        };
        self.backend.launch(
            Kernel::RopeKv(shape.kv_precision),
            &rope,
            &[
                Binding::whole(1, &a.qkv),
                Binding::whole(2, self.inputs.cosines),
                Binding::whole(3, self.inputs.sines),
                Binding::whole(4, &a.tokens),
                Binding::Table {
                    index: 5,
                    table: self.inputs.table,
                },
            ],
            [
                1,
                (shape.heads + 2 * shape.kv_heads) as usize,
                count as usize,
            ],
            [(HEAD_DIM / 2) as usize, 1, 1],
        )
    }

    pub(super) fn feed_forward_half(
        &mut self,
        offsets: &DecoderLayerOffsets,
    ) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        let ffn = shape.feed_forward;
        let count = self.span.0;
        self.norm(offsets.feed_forward_norm)?;
        let gate_up = [
            self.segment(offsets.gate, NO_BIAS, ffn, 0),
            self.segment(offsets.up, NO_BIAS, ffn, ffn),
        ];
        self.project(
            &gate_up,
            shape.hidden,
            [&a.normalized, &a.gate_up],
            [shape.hidden, 2 * ffn],
            false,
        )?;
        let swiglu = SwigluArgs {
            columns: ffn,
            tokens: count,
            in_stride: 2 * ffn,
            out_stride: ffn,
        };
        self.backend.launch(
            Kernel::Swiglu,
            &swiglu,
            &[
                Binding::whole(1, &a.gate_up),
                Binding::whole(2, &a.feed_forward),
            ],
            [(ffn as usize).div_ceil(ELEMENT_THREADS), count as usize, 1],
            [ELEMENT_THREADS, 1, 1],
        )?;
        let down = [self.segment(offsets.down, NO_BIAS, shape.hidden, 0)];
        self.project(
            &down,
            ffn,
            [&a.feed_forward, &a.hidden],
            [ffn, shape.hidden],
            true,
        )
    }

    pub(super) fn logits(&mut self, rows: u32) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        self.normalize(self.inputs.plan.output_norm, rows, true, &a.logits_input)?;
        let output = [self.segment(self.inputs.plan.output, NO_BIAS, shape.vocabulary, 0)];
        project(
            self.backend,
            self.inputs.weights,
            &Projection {
                segments: &output,
                columns: shape.hidden,
                span: (rows, rows),
                strides: [shape.hidden, shape.vocabulary],
                accumulate: false,
            },
            [&a.logits_input, &a.logits],
        )
    }

    fn norm(&mut self, weight: u64) -> Result<(), GpuError> {
        let a = self.inputs.activations;
        self.normalize(weight, self.span.0, false, &a.normalized)
    }

    /// RMS-normalizes `rows` hidden rows into `output`; in gather mode row r reads the r-th logits token.
    fn normalize(
        &mut self,
        weight: u64,
        rows: u32,
        gather: bool,
        output: &S::Buffer,
    ) -> Result<(), GpuError> {
        let shape = self.inputs.plan.shape;
        let a = self.inputs.activations;
        let norm = NormArgs {
            weight: self.weight(weight),
            columns: shape.hidden,
            rows,
            in_stride: shape.hidden,
            out_stride: shape.hidden,
            epsilon: shape.rms_epsilon,
            gather_logits: u32::from(gather),
            token_count: self.span.0,
            pad: 0,
        };
        self.backend.launch(
            Kernel::RmsNorm,
            &norm,
            &[
                Binding::whole(1, self.inputs.weights),
                Binding::whole(2, &a.hidden),
                Binding::whole(3, output),
                Binding::whole(4, &a.gather),
            ],
            [rows as usize, 1, 1],
            [NORM_THREADS, 1, 1],
        )
    }

    fn project(
        &mut self,
        segments: &[MatmulSegment],
        columns: u32,
        buffers: [&S::Buffer; 2],
        strides: [u32; 2],
        accumulate: bool,
    ) -> Result<(), GpuError> {
        let projection = Projection {
            segments,
            columns,
            span: self.span,
            strides,
            accumulate,
        };
        project(self.backend, self.inputs.weights, &projection, buffers)
    }

    const fn segment(&self, weight: u64, bias: u64, rows: u32, out_column: u32) -> MatmulSegment {
        MatmulSegment {
            weight: self.weight(weight),
            bias: if bias == NO_BIAS {
                NO_BIAS
            } else {
                self.weight(bias)
            },
            rows,
            out_column,
        }
    }
}
