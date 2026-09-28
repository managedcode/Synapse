//! A dense decoder on CUDA device 0: device copies of the weights and tables, lazily allocated KV slots, and
//! the shared step encoding launched on the default stream.

use std::ffi::c_void;

use super::CudaContext;
use super::driver::{CuDevicePtr, CuFunction};
use crate::gpu_kernels::GpuDeviceInfo;
use crate::gpu_kernels::decoder::{BatchToken, DecoderPlan, KvPrecision};
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::params::{MAX_SLOTS, SlotTable};
use crate::gpu_kernels::schedule::plan_attention;
use crate::gpu_kernels::step::{
    Activations, Binding, Kernel, StepBackend, StepInputs, encode_step,
};

/// One attention tile of FP32 keys after the last value position (as on Metal).
const SLOT_PADDING_BYTES: u64 = 32 * 64 * 4;

/// A device allocation; freed by the owning decoder.
pub struct DeviceBuffer {
    pointer: CuDevicePtr,
}

/// A loaded dense decoder on CUDA device 0.
pub struct CudaDecoder {
    context: CudaContext,
    plan: DecoderPlan,
    weights: DeviceBuffer,
    cosines: DeviceBuffer,
    sines: DeviceBuffer,
    activations: Activations<DeviceBuffer>,
    slots: Vec<Option<DeviceBuffer>>,
}

impl CudaDecoder {
    /// Copies the mapped weights and `RoPE` tables to the device.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] without a usable CUDA device, [`GpuError::OutOfMemory`] when the
    /// device cannot hold the model, and [`GpuError::Device`] for other driver failures.
    pub fn new(
        plan: DecoderPlan,
        weights: &[u8],
        cosines: &[f32],
        sines: &[f32],
    ) -> Result<Self, GpuError> {
        if plan.shape.kv_precision != KvPrecision::F32 {
            return Err(GpuError::Unavailable(
                "the FP16 KV profile is implemented for Metal only; CUDA uses FP32 KV".into(),
            ));
        }

        if cosines.len() != plan.rope_floats || sines.len() != plan.rope_floats {
            return Err(GpuError::invalid(
                "RoPE tables must hold context * head_dim / 2 floats",
            ));
        }

        let context = CudaContext::new()?;
        let weights_buffer = allocate(&context, weights.len() as u64, "weights")?;
        copy_to_device(&context, &weights_buffer, weights)?;
        let cosine_buffer = allocate(&context, (cosines.len() * 4) as u64, "rope cosines")?;
        copy_to_device(&context, &cosine_buffer, cosines)?;
        let sine_buffer = allocate(&context, (sines.len() * 4) as u64, "rope sines")?;
        copy_to_device(&context, &sine_buffer, sines)?;
        let activations =
            Activations::sizes(&plan).try_map(|name, bytes| allocate(&context, bytes, name))?;
        Ok(Self {
            slots: std::iter::repeat_with(|| None)
                .take(plan.shape.session_slots as usize)
                .collect(),
            context,
            plan,
            weights: weights_buffer,
            cosines: cosine_buffer,
            sines: sine_buffer,
            activations,
        })
    }

    /// The device this model runs on.
    #[must_use]
    pub const fn device_info(&self) -> &GpuDeviceInfo {
        &self.context.info
    }

    /// Evaluates one batched step and copies each requested logits row into `logits`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidArgument`] for invalid tokens or a short logits buffer and driver errors
    /// otherwise.
    pub fn forward(&mut self, tokens: &[BatchToken], logits: &mut [f32]) -> Result<(), GpuError> {
        let rows = self.plan.validate_step(tokens)?;
        let vocabulary = self.plan.shape.vocabulary as usize;
        if logits.len() < rows as usize * vocabulary {
            return Err(GpuError::invalid(
                "the logits buffer is shorter than rows * vocabulary",
            ));
        }

        let table = self.prepare_slots(tokens)?;
        copy_to_device(&self.context, &self.activations.tokens, tokens)?;
        let gather: Vec<u32> = (0..tokens.len())
            .filter(|&index| tokens[index].logits_row >= 0)
            .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
            .collect();
        copy_to_device(&self.context, &self.activations.gather, &gather)?;
        let attention = plan_attention(tokens, self.plan.shape.kv_heads);
        copy_to_device(&self.context, &self.activations.blocks, &attention.blocks)?;
        let inputs = StepInputs {
            plan: &self.plan,
            weights: &self.weights,
            weights_offset: 0,
            cosines: &self.cosines,
            sines: &self.sines,
            activations: &self.activations,
            table: &table,
            attention: &attention,
        };
        encode_step(
            &mut Launcher {
                context: &self.context,
            },
            &inputs,
            tokens.len(),
            rows,
        )?;
        let driver = &self.context.driver;
        // SAFETY: synchronizing the current context has no arguments.
        let synchronized = unsafe { (driver.context_synchronize)() };
        driver.check(synchronized, "cuCtxSynchronize")?;
        let destination = &mut logits[..rows as usize * vocabulary];
        // SAFETY: the logits buffer holds at least `destination.len()` floats and the device is idle.
        let copied = unsafe {
            (driver.copy_to_host)(
                destination.as_mut_ptr().cast::<c_void>(),
                self.activations.logits.pointer,
                size_of_val(destination),
            )
        };
        driver.check(copied, "cuMemcpyDtoH")
    }

    fn prepare_slots(&mut self, tokens: &[BatchToken]) -> Result<SlotTable, GpuError> {
        let bytes = self.plan.shape.slot_bytes() + SLOT_PADDING_BYTES;
        for token in tokens {
            let slot = token.slot.unsigned_abs() as usize;
            if self.slots[slot].is_none() {
                let buffer = allocate(&self.context, bytes, "KV slot")?;
                let driver = &self.context.driver;
                let length = usize::try_from(bytes).unwrap_or(usize::MAX);
                // SAFETY: the allocation holds `bytes` bytes.
                let cleared = unsafe { (driver.memset)(buffer.pointer, 0, length) };
                driver.check(cleared, "cuMemsetD8")?;
                self.slots[slot] = Some(buffer);
            }
        }

        let mut table = SlotTable {
            address: [0; MAX_SLOTS],
        };
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(buffer) = slot {
                table.address[index] = buffer.pointer;
            }
        }

        Ok(table)
    }
}

impl Drop for CudaDecoder {
    fn drop(&mut self) {
        let activations = &self.activations;
        let buffers = [
            &self.weights,
            &self.cosines,
            &self.sines,
            &activations.tokens,
            &activations.hidden,
            &activations.normalized,
            &activations.qkv,
            &activations.attention,
            &activations.gate_up,
            &activations.feed_forward,
            &activations.logits_input,
            &activations.logits,
            &activations.gather,
            &activations.blocks,
            &activations.partial,
        ];
        for buffer in buffers.into_iter().chain(self.slots.iter().flatten()) {
            // SAFETY: every pointer came from `cuMemAlloc` and is freed exactly once here.
            let _ = unsafe { (self.context.driver.memory_free)(buffer.pointer) };
        }
    }
}

fn allocate(context: &CudaContext, bytes: u64, name: &str) -> Result<DeviceBuffer, GpuError> {
    let length = usize::try_from(bytes.max(4))
        .map_err(|_| GpuError::OutOfMemory(format!("{name}: {bytes} bytes")))?;
    let mut pointer: CuDevicePtr = 0;
    let driver = &context.driver;
    // SAFETY: `pointer` is a live out-parameter.
    let allocated = unsafe { (driver.memory_allocate)(&raw mut pointer, length) };
    driver.check(allocated, name)?;
    Ok(DeviceBuffer { pointer })
}

fn copy_to_device<T: Copy>(
    context: &CudaContext,
    buffer: &DeviceBuffer,
    values: &[T],
) -> Result<(), GpuError> {
    if values.is_empty() {
        return Ok(());
    }

    let driver = &context.driver;
    // SAFETY: the device buffer was sized for these values and `values` is a live host slice.
    driver.check(
        unsafe {
            (driver.copy_to_device)(
                buffer.pointer,
                values.as_ptr().cast::<c_void>(),
                size_of_val(values),
            )
        },
        "cuMemcpyHtoD",
    )
}

/// Launches step kernels on the default stream; launches are ordered, so `flush` has nothing to do.
struct Launcher<'a> {
    context: &'a CudaContext,
}

impl Launcher<'_> {
    const fn function(&self, kernel: Kernel) -> CuFunction {
        let functions = &self.context.functions;
        match kernel {
            Kernel::Embed => functions.embed,
            Kernel::RmsNorm => functions.rms_norm,
            Kernel::Matvec(1) => functions.matvec[0],
            Kernel::Matvec(2) => functions.matvec[1],
            Kernel::Matvec(4) => functions.matvec[2],
            Kernel::Matvec(_) => functions.matvec[3],
            Kernel::Gemm => functions.gemm,
            Kernel::RopeKv(_) => functions.rope_kv,
            Kernel::Attention(_) => functions.attention,
            Kernel::AttentionDecode(_) => functions.attention_decode,
            Kernel::AttentionReduce => functions.attention_reduce,
            Kernel::Swiglu => functions.swiglu,
        }
    }
}

enum Parameter<'a> {
    Pointer(CuDevicePtr),
    Table(&'a SlotTable),
}

/// Kernel parameters follow the Metal buffer indices: the argument struct, then indices `1..=n` in order.
fn ordered_parameters<'a>(
    kernel: Kernel,
    bindings: &[Binding<'a, DeviceBuffer>],
) -> Result<Vec<Parameter<'a>>, GpuError> {
    let mut ordered: Vec<(usize, Parameter<'a>)> = bindings
        .iter()
        .map(|binding| match *binding {
            Binding::Buffer {
                index,
                buffer,
                offset,
            } => (index, Parameter::Pointer(buffer.pointer + offset as u64)),
            Binding::Table { index, table } => (index, Parameter::Table(table)),
        })
        .collect();
    ordered.sort_by_key(|(index, _)| *index);
    if ordered
        .iter()
        .enumerate()
        .any(|(position, (index, _))| *index != position + 1)
    {
        return Err(GpuError::invalid(format!(
            "{kernel:?} bindings are not contiguous from index 1"
        )));
    }

    Ok(ordered
        .into_iter()
        .map(|(_, parameter)| parameter)
        .collect())
}

impl StepBackend for Launcher<'_> {
    type Buffer = DeviceBuffer;

    fn launch<T: Copy>(
        &mut self,
        kernel: Kernel,
        args: &T,
        bindings: &[Binding<'_, DeviceBuffer>],
        groups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<(), GpuError> {
        let ordered = ordered_parameters(kernel, bindings)?;
        let pointers: Vec<CuDevicePtr> = ordered
            .iter()
            .map(|parameter| match parameter {
                Parameter::Pointer(pointer) => *pointer,
                Parameter::Table(_) => 0,
            })
            .collect();
        let mut parameters: Vec<*mut c_void> = Vec::with_capacity(ordered.len() + 1);
        parameters.push(std::ptr::from_ref(args).cast_mut().cast::<c_void>());
        for (position, parameter) in ordered.iter().enumerate() {
            parameters.push(match parameter {
                Parameter::Pointer(_) => std::ptr::from_ref(&pointers[position])
                    .cast_mut()
                    .cast::<c_void>(),
                Parameter::Table(table) => std::ptr::from_ref(*table).cast_mut().cast::<c_void>(),
            });
        }

        let dimension = |value: usize| {
            u32::try_from(value).map_err(|_| GpuError::invalid("launch dimension overflows u32"))
        };
        let driver = &self.context.driver;
        // SAFETY: every parameter pointer refers to a live value of the kernel's declared parameter type, in
        // declaration order, for the duration of the call (the driver copies them before returning).
        let result = unsafe {
            (driver.launch)(
                self.function(kernel),
                dimension(groups[0])?,
                dimension(groups[1])?,
                dimension(groups[2])?,
                dimension(threads[0])?,
                dimension(threads[1])?,
                dimension(threads[2])?,
                0,
                std::ptr::null_mut(),
                parameters.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        driver.check(result, "cuLaunchKernel")
    }

    fn flush(&mut self) -> Result<(), GpuError> {
        Ok(())
    }
}
