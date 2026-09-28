//! NVRTC, loaded dynamically, compiles the CUDA C kernels to PTX for the device's compute capability at load.

use std::ffi::{CString, c_char, c_int};

use libloading::Library;

use super::driver::symbol;
use crate::gpu_kernels::error::GpuError;

type NvrtcResult = c_int;
type Program = *mut std::ffi::c_void;
type SizeFn = unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult;
type CopyFn = unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult;
type CreateFn = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;

#[cfg(target_os = "windows")]
const NVRTC_NAMES: &[&str] = &[
    "nvrtc64_130_0.dll",
    "nvrtc64_120_0.dll",
    "nvrtc64_112_0.dll",
];
#[cfg(not(target_os = "windows"))]
const NVRTC_NAMES: &[&str] = &[
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "libnvrtc.so",
];

/// Compiles `source` to NUL-terminated PTX for `compute_<major><minor>`.
///
/// # Errors
///
/// Returns [`GpuError::Unavailable`] without NVRTC and [`GpuError::Device`] with the compiler log on failure.
pub fn compile(source: &str, major: i32, minor: i32) -> Result<Vec<u8>, GpuError> {
    let nvrtc = Nvrtc::load()?;
    let source =
        CString::new(source).map_err(|_| GpuError::invalid("kernel source contains NUL"))?;
    let options: Vec<CString> = [
        format!("--gpu-architecture=compute_{major}{minor}"),
        "--std=c++17".into(),
    ]
    .into_iter()
    .map(|option| CString::new(option).unwrap_or_default())
    .collect();
    let option_pointers: Vec<*const c_char> =
        options.iter().map(|option| option.as_ptr()).collect();
    let mut program: Program = std::ptr::null_mut();
    // SAFETY: every pointer is a live NUL-terminated string, or null with a zero count, as NVRTC requires.
    let created = unsafe {
        (nvrtc.create)(
            &raw mut program,
            source.as_ptr(),
            c"synapse.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if created != 0 {
        return Err(GpuError::Device(format!(
            "nvrtcCreateProgram failed with {created}"
        )));
    }

    let count = c_int::try_from(option_pointers.len()).unwrap_or(0);
    // SAFETY: `program` was created above and the option pointers stay alive for the call.
    let compiled = unsafe { (nvrtc.compile)(program, count, option_pointers.as_ptr()) };
    let result = if compiled == 0 {
        // SAFETY: `program` is live and the functions are the matching PTX size/copy pair.
        unsafe { read(program, nvrtc.ptx_size, nvrtc.ptx) }
    } else {
        // SAFETY: as above, for the compiler log.
        let log = unsafe { read(program, nvrtc.log_size, nvrtc.log) }.unwrap_or_default();
        Err(GpuError::Device(format!(
            "NVRTC failed to compile the CUDA kernels: {}",
            String::from_utf8_lossy(&log)
        )))
    };
    // SAFETY: `program` is valid and destroyed exactly once.
    unsafe { (nvrtc.destroy)(&raw mut program) };
    result
}

/// NVRTC entry points; `_library` keeps them valid.
struct Nvrtc {
    _library: Library,
    create: CreateFn,
    compile: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> NvrtcResult,
    log_size: SizeFn,
    log: CopyFn,
    ptx_size: SizeFn,
    ptx: CopyFn,
    destroy: unsafe extern "C" fn(*mut Program) -> NvrtcResult,
}

impl Nvrtc {
    fn load() -> Result<Self, GpuError> {
        // SAFETY: loading NVRTC runs its initializers, which is the documented way to use it.
        let library = NVRTC_NAMES
            .iter()
            .find_map(|name| unsafe { Library::new(*name) }.ok())
            .ok_or_else(|| {
                GpuError::Unavailable("the NVRTC runtime compiler library is not installed".into())
            })?;
        // SAFETY: each symbol is resolved with the exact C signature documented in `nvrtc.h`.
        unsafe {
            Ok(Self {
                create: symbol(&library, b"nvrtcCreateProgram\0")?,
                compile: symbol(&library, b"nvrtcCompileProgram\0")?,
                log_size: symbol(&library, b"nvrtcGetProgramLogSize\0")?,
                log: symbol(&library, b"nvrtcGetProgramLog\0")?,
                ptx_size: symbol(&library, b"nvrtcGetPTXSize\0")?,
                ptx: symbol(&library, b"nvrtcGetPTX\0")?,
                destroy: symbol(&library, b"nvrtcDestroyProgram\0")?,
                _library: library,
            })
        }
    }
}

/// Reads a sized NVRTC output (PTX or log) into a byte vector.
///
/// # Safety
///
/// `program` must be a live NVRTC program and the two functions must be the matching size/copy pair.
unsafe fn read(program: Program, size: SizeFn, copy: CopyFn) -> Result<Vec<u8>, GpuError> {
    let mut length = 0_usize;
    // SAFETY: forwarded from this function's contract.
    if unsafe { size(program, &raw mut length) } != 0 {
        return Err(GpuError::Device(
            "NVRTC could not report an output size".into(),
        ));
    }

    let mut bytes = vec![0_u8; length.max(1)];
    // SAFETY: `bytes` holds at least `length` bytes.
    if unsafe { copy(program, bytes.as_mut_ptr().cast::<c_char>()) } != 0 {
        return Err(GpuError::Device("NVRTC could not copy its output".into()));
    }

    Ok(bytes)
}
