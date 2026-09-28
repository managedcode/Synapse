//! GPU kernels slice: model description, device probing, and the C ABI.

pub mod decoder;
pub mod error;
#[allow(unsafe_code)]
pub mod ffi;
pub mod params;
pub mod schedule;
pub mod step;

#[allow(unsafe_code)]
pub mod cuda;
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub mod metal;

/// Device facts reported to the managed caller.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuDeviceInfo {
    /// NUL-terminated UTF-8 device name.
    pub name: [u8; 128],
    /// Memory the device can use without degrading performance.
    pub recommended_working_set_bytes: u64,
    /// Largest single buffer.
    pub max_buffer_bytes: u64,
    /// 1 when CPU and GPU share memory.
    pub unified_memory: u32,
    /// Highest supported Apple GPU family (7 = M1/A14), or 0.
    pub apple_family: u32,
}

impl GpuDeviceInfo {
    pub(crate) fn with_name(name: &str) -> Self {
        let mut bytes = [0_u8; 128];
        let mut length = name.len().min(bytes.len() - 1);
        while !name.is_char_boundary(length) {
            length -= 1;
        }

        bytes[..length].copy_from_slice(&name.as_bytes()[..length]);
        Self {
            name: bytes,
            recommended_working_set_bytes: 0,
            max_buffer_bytes: 0,
            unified_memory: 0,
            apple_family: 0,
        }
    }
}
