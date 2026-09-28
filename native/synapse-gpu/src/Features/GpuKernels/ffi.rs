//! The versioned C ABI (ADR-012). Every entry point validates pointers, catches panics, returns a status
//! code, and records a message that `synapse_gpu_last_error` copies out for the managed caller.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::GpuDeviceInfo;
use super::decoder::{BatchToken, DecoderDesc, DecoderLayerOffsets, DecoderPlan};
use super::error::{GpuError, STATUS_OK, STATUS_PANIC};

/// C ABI version checked by the managed loader before any other call.
pub const ABI_VERSION: u32 = 1;
/// Apple Metal.
pub const BACKEND_METAL: u32 = 1;
/// NVIDIA CUDA.
pub const BACKEND_CUDA: u32 = 2;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// An opaque loaded model owned by the managed caller.
pub enum GpuModel {
    #[cfg(target_os = "macos")]
    Metal(super::metal::decoder::MetalDecoder),
    Cuda(super::cuda::decoder::CudaDecoder),
}

fn record(error: &GpuError) -> i32 {
    LAST_ERROR.with(|last| *last.borrow_mut() = error.to_string());
    error.status()
}

fn guarded(body: impl FnOnce() -> Result<(), GpuError>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => STATUS_OK,
        Ok(Err(error)) => record(&error),
        Err(_) => {
            LAST_ERROR
                .with(|last| *last.borrow_mut() = "a Rust panic was caught at the GPU ABI".into());
            STATUS_PANIC
        }
    }
}

fn unsupported_backend(backend: u32) -> GpuError {
    match backend {
        BACKEND_METAL => GpuError::Unavailable("the Metal backend exists only on macOS".into()),
        other => GpuError::invalid(format!("unknown GPU backend {other}")),
    }
}

/// Returns [`ABI_VERSION`].
#[unsafe(no_mangle)]
pub const extern "C" fn synapse_gpu_abi_version() -> u32 {
    ABI_VERSION
}

/// Describes the backend's default device.
///
/// # Safety
///
/// `info` must be null or valid for one `GpuDeviceInfo` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn synapse_gpu_probe(backend: u32, info: *mut GpuDeviceInfo) -> i32 {
    if info.is_null() {
        return record(&GpuError::NullPointer("device info"));
    }

    guarded(|| {
        let described = probe(backend)?;
        // SAFETY: `info` is non-null and valid for one write per this function's contract.
        unsafe { info.write(described) };
        Ok(())
    })
}

fn probe(backend: u32) -> Result<GpuDeviceInfo, GpuError> {
    match backend {
        #[cfg(target_os = "macos")]
        BACKEND_METAL => super::metal::probe(),
        BACKEND_CUDA => super::cuda::probe(),
        other => Err(unsupported_backend(other)),
    }
}

/// Copies the calling thread's last error message (UTF-8, not terminated) and returns its full length.
///
/// # Safety
///
/// `buffer` must be null or valid for `capacity` byte writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn synapse_gpu_last_error(buffer: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|last| {
        let message = last.borrow();
        if !buffer.is_null() {
            let length = message.len().min(capacity);
            // SAFETY: `buffer` is valid for `capacity >= length` bytes and cannot alias the Rust string.
            unsafe { std::ptr::copy_nonoverlapping(message.as_ptr(), buffer, length) };
        }

        message.len()
    })
}

/// Loads a dense decoder on `backend` from caller-owned mapped weights.
///
/// # Safety
///
/// `desc` must point at a valid description whose pointers stay valid as documented on [`DecoderDesc`]; the
/// weight mapping must outlive the model. `model` must be valid for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn synapse_gpu_decoder_create(
    backend: u32,
    desc: *const DecoderDesc,
    model: *mut *mut GpuModel,
) -> i32 {
    if desc.is_null() || model.is_null() {
        return record(&GpuError::NullPointer("model description or output"));
    }

    guarded(|| {
        // SAFETY: `desc` is non-null and valid per this function's contract.
        let desc = unsafe { &*desc };
        if desc.layers.is_null()
            || desc.weights.is_null()
            || desc.rope_cosines.is_null()
            || desc.rope_sines.is_null()
        {
            return Err(GpuError::NullPointer("model description"));
        }

        // SAFETY: the caller guarantees `layer_count` layer entries at `layers`.
        let layers: &[DecoderLayerOffsets] =
            unsafe { std::slice::from_raw_parts(desc.layers, desc.layer_count as usize) };
        let plan = DecoderPlan::new(desc, layers)?;
        let loaded = create(backend, plan, desc)?;
        // SAFETY: `model` is non-null and valid for one write.
        unsafe { model.write(Box::into_raw(Box::new(loaded))) };
        Ok(())
    })
}

fn create(backend: u32, plan: DecoderPlan, desc: &DecoderDesc) -> Result<GpuModel, GpuError> {
    // SAFETY: the caller guarantees both tables hold `rope_floats` floats (validated length from `plan`).
    let (cosines, sines) = unsafe {
        (
            std::slice::from_raw_parts(desc.rope_cosines, plan.rope_floats),
            std::slice::from_raw_parts(desc.rope_sines, plan.rope_floats),
        )
    };
    match backend {
        #[cfg(target_os = "macos")]
        BACKEND_METAL => {
            let weights = std::ptr::NonNull::new(desc.weights.cast_mut())
                .ok_or(GpuError::NullPointer("weights"))?;
            // SAFETY: the caller keeps the weight mapping alive for the model's lifetime.
            let decoder =
                unsafe { super::metal::decoder::MetalDecoder::new(plan, weights, cosines, sines)? };
            Ok(GpuModel::Metal(decoder))
        }
        BACKEND_CUDA => {
            let length = usize::try_from(plan.weights_length)
                .map_err(|_| GpuError::invalid("weights length overflows usize"))?;
            // SAFETY: the caller guarantees `weights_length` readable bytes at `weights`.
            let weights = unsafe { std::slice::from_raw_parts(desc.weights, length) };
            let decoder = super::cuda::decoder::CudaDecoder::new(plan, weights, cosines, sines)?;
            Ok(GpuModel::Cuda(decoder))
        }
        other => Err(unsupported_backend(other)),
    }
}

/// Evaluates one batched step and writes `k * vocabulary` logits for its `k` logits rows.
///
/// # Safety
///
/// `model` must come from `synapse_gpu_decoder_create` and not be used concurrently; `tokens` must hold `count`
/// entries and `logits` `logits_capacity` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn synapse_gpu_decoder_forward(
    model: *mut GpuModel,
    tokens: *const BatchToken,
    count: usize,
    logits: *mut f32,
    logits_capacity: usize,
) -> i32 {
    if model.is_null() || tokens.is_null() || logits.is_null() {
        return record(&GpuError::NullPointer("model, tokens, or logits"));
    }

    guarded(|| {
        // SAFETY: pointers are non-null and valid for the stated lengths per this function's contract.
        let (model, tokens, logits) = unsafe {
            (
                &mut *model,
                std::slice::from_raw_parts(tokens, count),
                std::slice::from_raw_parts_mut(logits, logits_capacity),
            )
        };
        forward(model, tokens, logits)
    })
}

fn forward(
    model: &mut GpuModel,
    tokens: &[BatchToken],
    logits: &mut [f32],
) -> Result<(), GpuError> {
    match model {
        #[cfg(target_os = "macos")]
        GpuModel::Metal(decoder) => decoder.forward(tokens, logits),
        GpuModel::Cuda(decoder) => decoder.forward(tokens, logits),
    }
}

/// Releases a model; null is ignored.
///
/// # Safety
///
/// `model` must be null or come from `synapse_gpu_decoder_create` and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn synapse_gpu_decoder_destroy(model: *mut GpuModel) {
    if !model.is_null() {
        // SAFETY: the pointer came from `Box::into_raw` in `synapse_gpu_decoder_create`.
        drop(unsafe { Box::from_raw(model) });
    }
}
