//! Metal buffers and the [`StepBackend`] submission. Every `unsafe` block is an Objective-C call whose
//! arguments are live Rust values or retained Metal objects.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};

use super::{MetalContext, Pipeline};
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::step::{Binding, Kernel, StepBackend};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ComputeEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Allocates a shared (unified-memory) buffer of at least four bytes.
pub fn shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: u64,
    name: &str,
) -> Result<Buffer, GpuError> {
    let length = usize::try_from(bytes.max(4))
        .map_err(|_| GpuError::OutOfMemory(format!("{name}: {bytes} bytes")))?;
    device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| GpuError::OutOfMemory(format!("{name}: {bytes} bytes")))
}

/// Copies `values` to the start of a shared buffer.
pub fn upload<T: Copy>(buffer: &Buffer, values: &[T]) {
    let bytes = size_of_val(values);
    assert!(bytes <= buffer.length(), "upload exceeds the buffer");
    // SAFETY: `contents` is the CPU address of a shared buffer of at least `bytes` bytes, the GPU is idle
    // (every step waits for completion), and `values` cannot overlap device memory.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
}

/// Copies the first `destination.len()` floats of a shared buffer out.
pub fn download(buffer: &Buffer, destination: &mut [f32]) {
    let bytes = size_of_val(destination);
    assert!(bytes <= buffer.length(), "download exceeds the buffer");
    // SAFETY: as in `upload`, with the copy direction reversed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<u8>(),
            destination.as_mut_ptr().cast::<u8>(),
            bytes,
        );
    }
}

/// Fills a buffer with zeros on the GPU and waits, so no allocation exposes unspecified bytes to a kernel.
pub fn zero_fill(context: &MetalContext, buffer: &Buffer) -> Result<(), GpuError> {
    let command = new_command(context)?;
    let blit = command
        .blitCommandEncoder()
        .ok_or_else(|| GpuError::Device("the command buffer returned no blit encoder".into()))?;
    blit.fillBuffer_range_value(buffer, NSRange::new(0, buffer.length()), 0);
    blit.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    check(&command)
}

fn new_command(context: &MetalContext) -> Result<CommandBuffer, GpuError> {
    context
        .queue
        .commandBuffer()
        .ok_or_else(|| GpuError::Device("the queue returned no command buffer".into()))
}

fn check(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), GpuError> {
    if command.status() == MTLCommandBufferStatus::Error {
        let message = command
            .error()
            .map_or_else(|| "unknown error".to_owned(), |error| error.to_string());
        return Err(GpuError::Device(format!(
            "the Metal step failed: {message}"
        )));
    }

    Ok(())
}

/// One step's submissions: a current command buffer and compute encoder, plus those already committed.
pub struct Submission<'a> {
    context: &'a MetalContext,
    resident: Vec<&'a Buffer>,
    committed: Vec<CommandBuffer>,
    command: CommandBuffer,
    encoder: ComputeEncoder,
}

impl<'a> Submission<'a> {
    /// Opens a submission; `resident` buffers are reached through GPU addresses and declared per encoder.
    pub fn begin(context: &'a MetalContext, resident: Vec<&'a Buffer>) -> Result<Self, GpuError> {
        let (command, encoder) = Self::open(context, &resident)?;
        Ok(Self {
            context,
            resident,
            committed: Vec::new(),
            command,
            encoder,
        })
    }

    /// Commits the last command buffer, waits for all of them, reports the first failure, and returns the
    /// summed GPU seconds.
    pub fn finish(self) -> Result<f64, GpuError> {
        self.encoder.endEncoding();
        self.command.commit();
        self.command.waitUntilCompleted();
        let mut gpu_seconds = 0.0;
        for command in self.committed.iter().chain(std::iter::once(&self.command)) {
            command.waitUntilCompleted();
            check(command)?;
            gpu_seconds += command.GPUEndTime() - command.GPUStartTime();
        }

        Ok(gpu_seconds)
    }

    fn open(
        context: &MetalContext,
        resident: &[&Buffer],
    ) -> Result<(CommandBuffer, ComputeEncoder), GpuError> {
        let command = new_command(context)?;
        let encoder = command.computeCommandEncoder().ok_or_else(|| {
            GpuError::Device("the command buffer returned no compute encoder".into())
        })?;
        for buffer in resident {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***buffer);
            encoder.useResource_usage(resource, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }

        Ok((command, encoder))
    }

    fn pipeline(&self, kernel: Kernel) -> &Pipeline {
        let pipelines = &self.context.pipelines;
        match kernel {
            Kernel::Embed => &pipelines.embed,
            Kernel::RmsNorm => &pipelines.rms_norm,
            Kernel::Matvec(1) => &pipelines.matvec[0],
            Kernel::Matvec(2) => &pipelines.matvec[1],
            Kernel::Matvec(4) => &pipelines.matvec[2],
            Kernel::Matvec(_) => &pipelines.matvec[3],
            Kernel::Gemm => &pipelines.gemm,
            Kernel::RopeKv(precision) => &pipelines.rope_kv[precision as usize],
            Kernel::Attention(precision) => &pipelines.attention[precision as usize],
            Kernel::AttentionDecode(precision) => &pipelines.attention_decode[precision as usize],
            Kernel::AttentionReduce => &pipelines.attention_reduce,
            Kernel::Swiglu => &pipelines.swiglu,
        }
    }

    fn bytes<T: Copy>(&self, value: &T, index: usize) {
        let pointer = NonNull::from(value).cast::<c_void>();
        // SAFETY: `value` is a live `Copy` value of `size_of::<T>()` bytes (< 4 KiB); Metal copies it now.
        unsafe {
            self.encoder
                .setBytes_length_atIndex(pointer, size_of::<T>(), index);
        }
    }
}

impl StepBackend for Submission<'_> {
    type Buffer = Buffer;

    fn launch<T: Copy>(
        &mut self,
        kernel: Kernel,
        args: &T,
        bindings: &[Binding<'_, Buffer>],
        groups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<(), GpuError> {
        self.encoder.setComputePipelineState(self.pipeline(kernel));
        self.bytes(args, 0);
        for binding in bindings {
            match *binding {
                Binding::Buffer {
                    index,
                    buffer,
                    offset,
                } => {
                    // SAFETY: the buffer is retained by the model for the whole step, the offset lies inside
                    // it, and every kernel bounds its accesses by arguments validated in `DecoderPlan`.
                    unsafe {
                        self.encoder
                            .setBuffer_offset_atIndex(Some(buffer), offset, index);
                    }
                }
                Binding::Table { index, table } => self.bytes(table, index),
            }
        }

        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
        Ok(())
    }

    fn flush(&mut self) -> Result<(), GpuError> {
        let (command, encoder) = Self::open(self.context, &self.resident)?;
        self.encoder.endEncoding();
        self.command.commit();
        self.committed
            .push(std::mem::replace(&mut self.command, command));
        self.encoder = encoder;
        Ok(())
    }
}

const fn size(values: [usize; 3]) -> MTLSize {
    MTLSize {
        width: values[0],
        height: values[1],
        depth: values[2],
    }
}
