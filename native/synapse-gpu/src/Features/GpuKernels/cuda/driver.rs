//! The CUDA driver API, loaded dynamically. A machine without the NVIDIA driver reports `Unavailable`; the
//! build never links against CUDA.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use libloading::Library;

use crate::gpu_kernels::error::GpuError;

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_OUT_OF_MEMORY: CuResult = 2;
pub const ATTRIBUTE_COMPUTE_MAJOR: c_int = 75;
pub const ATTRIBUTE_COMPUTE_MINOR: c_int = 76;

#[cfg(target_os = "windows")]
const DRIVER_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(not(target_os = "windows"))]
const DRIVER_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

/// Function pointers of the driver entry points Synapse uses. `_library` keeps them valid.
#[allow(clippy::struct_field_names)]
pub struct Driver {
    _library: Library,
    pub init: unsafe extern "C" fn(c_uint) -> CuResult,
    pub device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    pub device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    pub device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    pub device_total_memory: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    pub device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    pub primary_context_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    pub primary_context_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    pub context_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    pub context_synchronize: unsafe extern "C" fn() -> CuResult,
    pub module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    pub module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    pub module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    pub memory_allocate: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    pub memory_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    pub copy_to_device: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    pub copy_to_host: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    pub memset: unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    pub launch: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    pub error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

impl Driver {
    /// Loads the driver library and resolves every entry point.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] when no driver library loads or an entry point is missing.
    pub fn load() -> Result<Self, GpuError> {
        // SAFETY: loading the NVIDIA driver runs its initializers, which is the documented way to use it.
        let library = DRIVER_NAMES
            .iter()
            .find_map(|name| unsafe { Library::new(*name) }.ok())
            .ok_or_else(|| {
                GpuError::Unavailable("the NVIDIA CUDA driver library is not installed".into())
            })?;
        // SAFETY: each symbol is resolved with the exact C signature documented in `cuda.h`.
        unsafe {
            Ok(Self {
                init: symbol(&library, b"cuInit\0")?,
                device_get_count: symbol(&library, b"cuDeviceGetCount\0")?,
                device_get: symbol(&library, b"cuDeviceGet\0")?,
                device_get_name: symbol(&library, b"cuDeviceGetName\0")?,
                device_total_memory: symbol(&library, b"cuDeviceTotalMem_v2\0")?,
                device_get_attribute: symbol(&library, b"cuDeviceGetAttribute\0")?,
                primary_context_retain: symbol(&library, b"cuDevicePrimaryCtxRetain\0")?,
                primary_context_release: symbol(&library, b"cuDevicePrimaryCtxRelease_v2\0")?,
                context_set_current: symbol(&library, b"cuCtxSetCurrent\0")?,
                context_synchronize: symbol(&library, b"cuCtxSynchronize\0")?,
                module_load_data: symbol(&library, b"cuModuleLoadData\0")?,
                module_unload: symbol(&library, b"cuModuleUnload\0")?,
                module_get_function: symbol(&library, b"cuModuleGetFunction\0")?,
                memory_allocate: symbol(&library, b"cuMemAlloc_v2\0")?,
                memory_free: symbol(&library, b"cuMemFree_v2\0")?,
                copy_to_device: symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                copy_to_host: symbol(&library, b"cuMemcpyDtoH_v2\0")?,
                memset: symbol(&library, b"cuMemsetD8_v2\0")?,
                launch: symbol(&library, b"cuLaunchKernel\0")?,
                error_string: symbol(&library, b"cuGetErrorString\0")?,
                _library: library,
            })
        }
    }

    /// Converts a driver result into a typed error that names the failing call.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::OutOfMemory`] for allocation failures and [`GpuError::Device`] otherwise.
    pub fn check(&self, result: CuResult, call: &str) -> Result<(), GpuError> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }

        let mut text: *const c_char = std::ptr::null();
        // SAFETY: `cuGetErrorString` writes a pointer to a static NUL-terminated string or leaves null.
        let described = unsafe { (self.error_string)(result, &raw mut text) } == CUDA_SUCCESS
            && !text.is_null();
        let message = if described {
            // SAFETY: the driver returned a valid static C string.
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("CUDA error {result}")
        };
        Err(if result == CUDA_ERROR_OUT_OF_MEMORY {
            GpuError::OutOfMemory(format!("{call}: {message}"))
        } else {
            GpuError::Device(format!("{call}: {message}"))
        })
    }
}

/// Resolves one symbol and copies the function pointer out of the borrowed `Symbol`.
///
/// # Safety
///
/// `T` must be the symbol's exact function-pointer type; the library must outlive the pointer.
pub unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, GpuError> {
    // SAFETY: forwarded from this function's contract.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            GpuError::Unavailable(format!(
                "the CUDA library lacks {}: {error}",
                String::from_utf8_lossy(name.strip_suffix(b"\0").unwrap_or(name))
            ))
        })
}
