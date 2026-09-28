//! Apple Metal backend (ADR-012).
//!
//! Kernels are compiled from source at model load with fast math disabled, so the build needs no Metal
//! toolchain. Requires an Apple7+ GPU (simdgroup matrices) and macOS 15+.

pub mod decoder;
mod encoder;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary, MTLMathMode,
};

use super::GpuDeviceInfo;
use super::error::GpuError;

const SOURCE: &str = concat!(
    include_str!("shaders/common.metal"),
    include_str!("shaders/elementwise.metal"),
    include_str!("shaders/matmul.metal"),
    include_str!("shaders/attention.metal"),
);

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Returns the highest Apple GPU family the device supports, or 0.
fn apple_family(device: &ProtocolObject<dyn MTLDevice>) -> u32 {
    [
        (MTLGPUFamily::Apple9, 9),
        (MTLGPUFamily::Apple8, 8),
        (MTLGPUFamily::Apple7, 7),
    ]
    .into_iter()
    .find(|(family, _)| device.supportsFamily(*family))
    .map_or(0, |(_, number)| number)
}

fn system_device() -> Result<Device, GpuError> {
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| GpuError::Unavailable("no Metal device is present".into()))
}

/// Describes the system default Metal device.
///
/// # Errors
///
/// Returns [`GpuError::Unavailable`] when the machine has no Metal device.
pub fn probe() -> Result<GpuDeviceInfo, GpuError> {
    let device = system_device()?;
    let mut info = GpuDeviceInfo::with_name(&device.name().to_string());
    info.recommended_working_set_bytes = device.recommendedMaxWorkingSetSize();
    info.max_buffer_bytes = u64::try_from(device.maxBufferLength()).unwrap_or(u64::MAX);
    info.unified_memory = u32::from(device.hasUnifiedMemory());
    info.apple_family = apple_family(&device);
    Ok(info)
}

/// Compiled pipelines for every Synapse kernel.
pub struct Pipelines {
    pub embed: Pipeline,
    pub rms_norm: Pipeline,
    /// Matrix-vector kernels for 1, 2, 4, and 8 tokens per threadgroup.
    pub matvec: [Pipeline; 4],
    /// Tiled FP32 simdgroup-matrix GEMM for prompt runs.
    pub gemm: Pipeline,
    /// FP32 and FP16 KV variants, indexed by `KvPrecision`.
    pub rope_kv: [Pipeline; 2],
    pub attention: [Pipeline; 2],
    pub attention_decode: [Pipeline; 2],
    pub attention_reduce: Pipeline,
    pub swiglu: Pipeline,
}

/// A device, its command queue, and the compiled kernels.
pub struct MetalContext {
    pub device: Device,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pipelines: Pipelines,
    pub info: GpuDeviceInfo,
}

impl MetalContext {
    /// Opens the default device and compiles the kernels.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] for a missing or pre-Apple7 device and [`GpuError::Device`] when
    /// compilation fails.
    pub fn new() -> Result<Self, GpuError> {
        let info = probe()?;
        let device = system_device()?;
        if info.apple_family < 7 || info.unified_memory == 0 {
            return Err(GpuError::Unavailable(format!(
                "the Metal backend needs an Apple7+ unified-memory GPU; this device reports family {}",
                info.apple_family
            )));
        }

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| GpuError::Device("the device returned no command queue".into()))?;
        let pipelines = compile_pipelines(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            info,
        })
    }
}

/// Compiles the kernel source with fast math disabled and builds every pipeline.
fn compile_pipelines(device: &ProtocolObject<dyn MTLDevice>) -> Result<Pipelines, GpuError> {
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    options.setLanguageVersion(MTLLanguageVersion::Version3_1);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
        .map_err(|error| GpuError::Device(format!("Metal kernel compilation failed: {error}")))?;
    let pipeline = |name: &str| -> Result<Pipeline, GpuError> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| {
                GpuError::Device(format!("kernel {name} is missing from the library"))
            })?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| GpuError::Device(format!("pipeline {name} failed: {error}")))
    };
    Ok(Pipelines {
        embed: pipeline("synapse_embed_q8_0")?,
        rms_norm: pipeline("synapse_rms_norm")?,
        matvec: [
            pipeline("synapse_q8_matvec_1")?,
            pipeline("synapse_q8_matvec_2")?,
            pipeline("synapse_q8_matvec_4")?,
            pipeline("synapse_q8_matvec_8")?,
        ],
        gemm: pipeline("synapse_q8_gemm")?,
        rope_kv: [
            pipeline("synapse_rope_kv")?,
            pipeline("synapse_rope_kv_f16")?,
        ],
        attention: [
            pipeline("synapse_attention")?,
            pipeline("synapse_attention_f16")?,
        ],
        attention_decode: [
            pipeline("synapse_attention_decode")?,
            pipeline("synapse_attention_decode_f16")?,
        ],
        attention_reduce: pipeline("synapse_attention_reduce")?,
        swiglu: pipeline("synapse_swiglu")?,
    })
}
