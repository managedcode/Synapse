//! NVIDIA CUDA backend (ADR-012).
//!
//! The driver API and NVRTC are loaded at runtime, so the build needs no CUDA toolkit and a machine without
//! the NVIDIA driver reports `Unavailable`. Kernels are compiled to PTX for the device at load. This backend
//! is not claimed until a CUDA device runs the parity suite.

mod admission;
pub mod decoder;
mod driver;
mod nvrtc;

use std::ffi::{CStr, c_char, c_int};

use driver::{
    ATTRIBUTE_COMPUTE_MAJOR, ATTRIBUTE_COMPUTE_MINOR, CuContext, CuDevice, CuFunction, CuModule,
    Driver,
};

use super::GpuDeviceInfo;
use super::error::GpuError;

const SOURCE: &str = concat!(
    include_str!("kernels/common.cuh"),
    include_str!("kernels/elementwise.cu"),
    include_str!("kernels/matmul.cu"),
    include_str!("kernels/attention.cu"),
);

/// Lowest compute capability the kernels target (warp shuffles with masks, `fmaf`, 48 KiB shared memory).
const MINIMUM_MAJOR: c_int = 6;

/// Describes CUDA device 0.
///
/// # Errors
///
/// Returns [`GpuError::Unavailable`] without a driver or device.
pub fn probe() -> Result<GpuDeviceInfo, GpuError> {
    let driver = Driver::load()?;
    Ok(describe(&driver)?.0)
}

fn describe(driver: &Driver) -> Result<(GpuDeviceInfo, CuDevice, c_int, c_int), GpuError> {
    // SAFETY: `cuInit(0)` has no pointer arguments.
    let initialized = unsafe { (driver.init)(0) };
    driver
        .check(initialized, "cuInit")
        .map_err(|error| GpuError::Unavailable(error.to_string()))?;
    let mut count = 0;
    // SAFETY: every out-parameter points at a live local of the documented type.
    unsafe {
        driver.check(
            (driver.device_get_count)(&raw mut count),
            "cuDeviceGetCount",
        )?;
    }
    if count < 1 {
        return Err(GpuError::Unavailable(
            "the CUDA driver reports no device".into(),
        ));
    }

    let (mut device, mut major, mut minor, mut memory) = (0, 0, 0, 0_usize);
    let mut name = [0 as c_char; 128];
    // SAFETY: as above; `name` holds 128 bytes and the driver NUL-terminates within that length.
    unsafe {
        driver.check((driver.device_get)(&raw mut device, 0), "cuDeviceGet")?;
        driver.check(
            (driver.device_get_name)(name.as_mut_ptr(), 128, device),
            "cuDeviceGetName",
        )?;
        driver.check(
            (driver.device_total_memory)(&raw mut memory, device),
            "cuDeviceTotalMem",
        )?;
        driver.check(
            (driver.device_get_attribute)(&raw mut major, ATTRIBUTE_COMPUTE_MAJOR, device),
            "cuDeviceGetAttribute",
        )?;
        driver.check(
            (driver.device_get_attribute)(&raw mut minor, ATTRIBUTE_COMPUTE_MINOR, device),
            "cuDeviceGetAttribute",
        )?;
    }
    // SAFETY: the driver wrote a NUL-terminated string into `name`.
    let text = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let mut info = GpuDeviceInfo::with_name(&format!("{text} sm_{major}{minor}"));
    info.recommended_working_set_bytes = memory as u64;
    info.max_buffer_bytes = memory as u64;
    Ok((info, device, major, minor))
}

/// Compiled functions for every Synapse kernel.
pub struct Functions {
    pub embed: CuFunction,
    pub rms_norm: CuFunction,
    pub matvec: [CuFunction; 4],
    pub gemm: CuFunction,
    pub rope_kv: CuFunction,
    pub attention: CuFunction,
    pub attention_decode: CuFunction,
    pub attention_reduce: CuFunction,
    pub swiglu: CuFunction,
}

/// The driver, the primary context of device 0, and the loaded kernel module.
pub struct CudaContext {
    pub driver: Driver,
    pub functions: Functions,
    pub info: GpuDeviceInfo,
    device: CuDevice,
    module: CuModule,
}

impl CudaContext {
    /// Opens device 0, compiles the kernels with NVRTC, and loads them.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] without a driver, device, or NVRTC, or for a device below compute
    /// capability 6.0, and [`GpuError::Device`] when compilation or loading fails.
    pub fn new() -> Result<Self, GpuError> {
        let driver = Driver::load()?;
        let (info, device, major, minor) = describe(&driver)?;
        if major < MINIMUM_MAJOR {
            return Err(GpuError::Unavailable(format!(
                "the CUDA kernels need compute capability 6.0+, found {major}.{minor}"
            )));
        }

        let mut context: CuContext = std::ptr::null_mut();
        // SAFETY: `context` is a live out-parameter and `device` came from the driver.
        unsafe {
            driver.check(
                (driver.primary_context_retain)(&raw mut context, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            driver.check((driver.context_set_current)(context), "cuCtxSetCurrent")?;
        }
        let ptx = nvrtc::compile(SOURCE, major, minor)?;
        let mut module: CuModule = std::ptr::null_mut();
        // SAFETY: `ptx` is NUL-terminated PTX produced by NVRTC for this device.
        let loaded = unsafe { (driver.module_load_data)(&raw mut module, ptx.as_ptr().cast()) };
        driver.check(loaded, "cuModuleLoadData")?;
        let function = |name: &CStr| -> Result<CuFunction, GpuError> {
            let mut function: CuFunction = std::ptr::null_mut();
            // SAFETY: `module` is loaded and `name` is NUL-terminated.
            let found =
                unsafe { (driver.module_get_function)(&raw mut function, module, name.as_ptr()) };
            driver.check(found, "cuModuleGetFunction")?;
            Ok(function)
        };
        let functions = Functions {
            embed: function(c"synapse_embed_q8_0")?,
            rms_norm: function(c"synapse_rms_norm")?,
            matvec: [
                function(c"synapse_q8_matvec_1")?,
                function(c"synapse_q8_matvec_2")?,
                function(c"synapse_q8_matvec_4")?,
                function(c"synapse_q8_matvec_8")?,
            ],
            gemm: function(c"synapse_q8_gemm")?,
            rope_kv: function(c"synapse_rope_kv")?,
            attention: function(c"synapse_attention")?,
            attention_decode: function(c"synapse_attention_decode")?,
            attention_reduce: function(c"synapse_attention_reduce")?,
            swiglu: function(c"synapse_swiglu")?,
        };
        Ok(Self {
            driver,
            functions,
            info,
            device,
            module,
        })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: the module and primary context were acquired in `new` and are released exactly once.
        unsafe {
            let _ = (self.driver.module_unload)(self.module);
            let _ = (self.driver.primary_context_release)(self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Driver, probe};
    use crate::gpu_kernels::error::GpuError;

    /// Without the NVIDIA driver the probe is `Unavailable`; with one it names a device. It never invents one.
    #[test]
    fn probe_reports_a_device_or_unavailable() {
        match (Driver::load(), probe()) {
            (Err(_), Err(GpuError::Unavailable(message))) => {
                assert!(message.contains("CUDA"), "{message}");
            }
            (Ok(_), Ok(info)) => assert!(info.name[0] != 0),
            (Ok(_), Err(GpuError::Unavailable(message))) => assert!(
                message.contains("device") || message.contains("cuInit"),
                "{message}"
            ),
            (driver, probed) => panic!(
                "inconsistent probe: driver {:?}, probe {probed:?}",
                driver.is_ok()
            ),
        }
    }
}
