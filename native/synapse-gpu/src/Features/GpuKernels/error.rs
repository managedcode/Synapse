use std::fmt;

/// The call completed.
pub const STATUS_OK: i32 = 0;
/// A required pointer was null.
pub const STATUS_NULL_POINTER: i32 = 1;
/// A shape, offset, token, or option was invalid.
pub const STATUS_INVALID_ARGUMENT: i32 = 2;
/// The requested backend or device is not available on this machine.
pub const STATUS_UNAVAILABLE: i32 = 3;
/// The device reported an error while compiling or executing.
pub const STATUS_DEVICE_ERROR: i32 = 4;
/// A device allocation failed.
pub const STATUS_OUT_OF_MEMORY: i32 = 5;
/// A Rust panic was caught before it reached the caller.
pub const STATUS_PANIC: i32 = 6;

/// An expected GPU backend failure with a message for the managed caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// A null pointer was passed across the ABI.
    NullPointer(&'static str),
    /// Invalid shape, offset, token, or option.
    InvalidArgument(String),
    /// The backend or device is missing or lacks a required feature.
    Unavailable(String),
    /// Shader compilation or command execution failed.
    Device(String),
    /// A device allocation failed.
    OutOfMemory(String),
}

impl GpuError {
    /// The ABI status code for this error.
    #[must_use]
    pub const fn status(&self) -> i32 {
        match self {
            Self::NullPointer(_) => STATUS_NULL_POINTER,
            Self::InvalidArgument(_) => STATUS_INVALID_ARGUMENT,
            Self::Unavailable(_) => STATUS_UNAVAILABLE,
            Self::Device(_) => STATUS_DEVICE_ERROR,
            Self::OutOfMemory(_) => STATUS_OUT_OF_MEMORY,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullPointer(name) => write!(formatter, "null pointer: {name}"),
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::Unavailable(message) => write!(formatter, "unavailable: {message}"),
            Self::Device(message) => write!(formatter, "device error: {message}"),
            Self::OutOfMemory(message) => write!(formatter, "out of device memory: {message}"),
        }
    }
}

impl std::error::Error for GpuError {}
