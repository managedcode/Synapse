//! A dense decoder on Metal: device buffers, lazily allocated KV slots, and the shared step encoding
//! (ADR-007 step semantics, ADR-012 numeric contract).

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::MetalContext;
use super::encoder::{
    Buffer, Submission, copy_regions, download, shared_buffer, upload, zero_fill,
};
use crate::gpu_kernels::GpuDeviceInfo;
use crate::gpu_kernels::decoder::{BatchToken, DecoderPlan, DecoderShape, HEAD_DIM};
use crate::gpu_kernels::error::GpuError;
use crate::gpu_kernels::params::{MAX_SLOTS, SlotTable};
use crate::gpu_kernels::schedule::{METAL_RUN_TOKENS, plan_attention};
use crate::gpu_kernels::step::{Activations, StepInputs, encode_step};

/// One attention tile of FP32 keys (32 x 64 floats) after the last value position.
const SLOT_PADDING_BYTES: u64 = 32 * 64 * 4;

unsafe extern "C" {
    fn getpagesize() -> i32;
}

/// One session's K and V and the positions its buffer holds, which is also the position stride (ADR-017).
struct KvSlot {
    buffer: Buffer,
    capacity: u32,
}

/// A loaded dense decoder on the default Metal device.
pub struct MetalDecoder {
    context: MetalContext,
    plan: DecoderPlan,
    weights: Buffer,
    weights_offset: u64,
    cosines: Buffer,
    sines: Buffer,
    activations: Activations<Buffer>,
    slots: Vec<Option<KvSlot>>,
}

impl MetalDecoder {
    /// Wraps the caller's mapped weights without copying and uploads the `RoPE` tables.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] without a suitable device, [`GpuError::Device`] when kernels fail to
    /// compile or the weights cannot be wrapped, and [`GpuError::OutOfMemory`] when a buffer cannot be created.
    ///
    /// # Safety
    ///
    /// `weights` must stay mapped and readable for `plan.weights_length` bytes until the model is dropped.
    pub unsafe fn new(
        plan: DecoderPlan,
        weights: NonNull<u8>,
        cosines: &[f32],
        sines: &[f32],
    ) -> Result<Self, GpuError> {
        if cosines.len() != plan.rope_floats || sines.len() != plan.rope_floats {
            return Err(GpuError::invalid(
                "RoPE tables must hold context * head_dim / 2 floats",
            ));
        }

        let context = MetalContext::new()?;
        // SAFETY: forwarded from this function's contract.
        let (weights, weights_offset) =
            unsafe { wrap_weights(&context, weights, plan.weights_length)? };
        let device = &context.device;
        let cosine_buffer = shared_buffer(device, bytes_of(cosines), "rope cosines")?;
        let sine_buffer = shared_buffer(device, bytes_of(sines), "rope sines")?;
        upload(&cosine_buffer, cosines);
        upload(&sine_buffer, sines);
        let activations =
            Activations::sizes(&plan).try_map(|name, bytes| shared_buffer(device, bytes, name))?;
        Ok(Self {
            slots: std::iter::repeat_with(|| None)
                .take(plan.shape.session_slots as usize)
                .collect(),
            context,
            plan,
            weights,
            weights_offset,
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
    /// Returns [`GpuError::InvalidArgument`] for invalid tokens or a short logits buffer,
    /// [`GpuError::OutOfMemory`] when a KV slot cannot be allocated, and [`GpuError::Device`] when a command
    /// buffer fails.
    pub fn forward(&mut self, tokens: &[BatchToken], logits: &mut [f32]) -> Result<(), GpuError> {
        let rows = self.plan.validate_step(tokens)?;
        let vocabulary = self.plan.shape.vocabulary as usize;
        if logits.len() < rows as usize * vocabulary {
            return Err(GpuError::invalid(
                "the logits buffer is shorter than rows * vocabulary",
            ));
        }

        let table = self.prepare_slots(tokens)?;
        upload(&self.activations.tokens, tokens);
        let gather: Vec<u32> = (0..tokens.len())
            .filter(|&index| tokens[index].logits_row >= 0)
            .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
            .collect();
        upload(&self.activations.gather, &gather);
        let attention = plan_attention(tokens, self.plan.shape.kv_heads, METAL_RUN_TOKENS);
        upload(&self.activations.blocks, &attention.blocks);

        let resident: Vec<&Buffer> = self
            .slots
            .iter()
            .flatten()
            .map(|slot| &slot.buffer)
            .collect();
        let mut submission = Submission::begin(&self.context, resident)?;
        let inputs = StepInputs {
            plan: &self.plan,
            weights: &self.weights,
            weights_offset: self.weights_offset,
            cosines: &self.cosines,
            sines: &self.sines,
            activations: &self.activations,
            table: &table,
            attention: &attention,
        };
        encode_step(&mut submission, &inputs, tokens.len(), rows)?;
        let gpu_seconds = submission.finish()?;
        if std::env::var_os("SYNAPSE_GPU_TRACE").is_some() {
            eprintln!(
                "synapse-gpu step tokens={} gpu_ms={:.3}",
                tokens.len(),
                gpu_seconds * 1e3
            );
        }

        download(
            &self.activations.logits,
            &mut logits[..rows as usize * vocabulary],
        );
        Ok(())
    }

    /// Sizes `slot` for `positions` at once (ADR-017), so a known request allocates one buffer instead of growing.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidArgument`] for an unknown slot or more positions than the context, and
    /// [`GpuError::OutOfMemory`] or [`GpuError::Device`] when the slot cannot be allocated or copied.
    pub fn reserve(&mut self, slot: usize, positions: u32) -> Result<(), GpuError> {
        if slot >= self.slots.len() || positions > self.plan.shape.context {
            return Err(GpuError::invalid(
                "reserve needs a known slot and at most the context positions",
            ));
        }

        let current = self.slots[slot].as_ref().map_or(0, |kv| kv.capacity);
        if positions > current {
            let capacity = self.plan.shape.reserve_capacity(positions);
            let grown = self.allocate_slot(capacity, self.slots[slot].as_ref())?;
            self.slots[slot] = Some(grown);
        }

        Ok(())
    }

    /// Bytes of K and V allocated across every slot, without padding.
    #[must_use]
    pub fn kv_bytes(&self) -> u64 {
        self.slots
            .iter()
            .flatten()
            .map(|slot| self.plan.shape.slot_bytes(slot.capacity))
            .sum()
    }

    /// Grows every slot the step writes beyond its capacity, then describes all slots for the kernels.
    fn prepare_slots(&mut self, tokens: &[BatchToken]) -> Result<SlotTable, GpuError> {
        let mut needed = vec![0_u32; self.slots.len()];
        for token in tokens {
            let slot = token.slot.unsigned_abs() as usize;
            needed[slot] = needed[slot].max(token.position.unsigned_abs() + 1);
        }

        for (index, &need) in needed.iter().enumerate() {
            let current = self.slots[index].as_ref().map_or(0, |slot| slot.capacity);
            if need > current {
                let grown = self.allocate_slot(
                    self.plan.shape.next_capacity(need, current),
                    self.slots[index].as_ref(),
                )?;
                self.slots[index] = Some(grown);
            }
        }

        let mut table = SlotTable {
            address: [0; MAX_SLOTS],
            capacity: [0; MAX_SLOTS],
            pad: 0,
        };
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(slot) = slot {
                table.address[index] = slot.buffer.gpuAddress();
                table.capacity[index] = slot.capacity;
            }
        }

        Ok(table)
    }

    /// A zero-filled slot of `capacity` positions holding a copy of `previous` (ADR-017). One tile of padding
    /// lets attention read a whole last tile; zero fill keeps unread keys finite.
    fn allocate_slot(&self, capacity: u32, previous: Option<&KvSlot>) -> Result<KvSlot, GpuError> {
        let shape = self.plan.shape;
        let buffer = shared_buffer(
            &self.context.device,
            shape.slot_bytes(capacity) + SLOT_PADDING_BYTES,
            "KV slot",
        )?;
        zero_fill(&self.context, &buffer)?;
        if let Some(previous) = previous {
            let regions = region_copies(&shape, previous.capacity, capacity);
            copy_regions(&self.context, &previous.buffer, &buffer, &regions)?;
        }

        Ok(KvSlot { buffer, capacity })
    }
}

/// `(source, target, length)` byte ranges that move every (K or V, layer, KV head) region of `from` positions
/// to its place in a slot of `to` positions; the layout is `[K|V][layer][KV head][position][HEAD_DIM]`.
fn region_copies(shape: &DecoderShape, from: u32, to: u32) -> Vec<(u64, u64, u64)> {
    let row = u64::from(HEAD_DIM) * shape.kv_precision.element_bytes();
    let regions = 2 * u64::from(shape.layer_count) * u64::from(shape.kv_heads);
    (0..regions)
        .map(|region| {
            (
                region * u64::from(from) * row,
                region * u64::from(to) * row,
                u64::from(from) * row,
            )
        })
        .collect()
}

/// Wraps the page-aligned span that covers `[weights, weights + length)` as a no-copy shared buffer.
///
/// # Safety
///
/// The whole page span must stay mapped and readable while the buffer lives.
unsafe fn wrap_weights(
    context: &MetalContext,
    weights: NonNull<u8>,
    length: u64,
) -> Result<(Buffer, u64), GpuError> {
    // SAFETY: `getpagesize` has no preconditions.
    let page = usize::try_from(unsafe { getpagesize() })
        .unwrap_or(16_384)
        .max(4_096);
    let address = weights.as_ptr() as usize;
    let base = address - address % page;
    let offset = address - base;
    let length =
        usize::try_from(length).map_err(|_| GpuError::invalid("weights length overflows usize"))?;
    let span = (offset + length).next_multiple_of(page);
    let pointer = NonNull::new(base as *mut c_void).ok_or(GpuError::NullPointer("weights"))?;
    // SAFETY: `base..base + span` lies inside the caller's page-aligned mapping per this function's contract,
    // and no deallocator is installed because the caller owns the mapping.
    let buffer = unsafe {
        context
            .device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                pointer,
                span,
                MTLResourceOptions::StorageModeShared,
                None,
            )
    };
    let buffer =
        buffer.ok_or_else(|| GpuError::Device("Metal could not wrap the mapped weights".into()))?;
    Ok((buffer, offset as u64))
}

const fn bytes_of(values: &[f32]) -> u64 {
    (values.len() * 4) as u64
}
