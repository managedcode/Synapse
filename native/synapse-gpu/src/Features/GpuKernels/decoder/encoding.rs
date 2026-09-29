//! Matrix encodings the kernels implement (ADR-021) and the per-layer packing of their GGML type IDs.

use crate::gpu_kernels::error::GpuError;

/// GGML type ID of `Q8_0`: 32 values in 34 bytes.
pub const ENCODING_Q8_0: u32 = 8;
/// GGML type ID of `Q4_K`: 256 values in 144 bytes.
pub const ENCODING_Q4_K: u32 = 12;
/// GGML type ID of `Q6_K`: 256 values in 210 bytes.
pub const ENCODING_Q6_K: u32 = 14;
/// Every matrix of a layer in `Q8_0`: seven four-bit fields of 8.
pub const ALL_Q8_0: u64 = 0x0888_8888;

/// One matrix encoding and its block geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Q8_0,
    Q4K,
    Q6K,
}

impl Encoding {
    /// The encoding with GGML type ID `id`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Unavailable`] for an encoding the kernels do not implement.
    pub fn from_id(id: u32) -> Result<Self, GpuError> {
        match id {
            ENCODING_Q8_0 => Ok(Self::Q8_0),
            ENCODING_Q4_K => Ok(Self::Q4K),
            ENCODING_Q6_K => Ok(Self::Q6K),
            other => Err(GpuError::Unavailable(format!(
                "the GPU kernels implement Q8_0, Q4_K, and Q6_K matrices, not GGML type {other}"
            ))),
        }
    }

    /// Values per block: 32 for `Q8_0`, 256 for the K-quants.
    #[must_use]
    pub const fn block_values(self) -> u64 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => 256,
        }
    }

    /// Bytes per block.
    #[must_use]
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }

    /// Bytes of a `rows` x `columns` matrix whose rows are a whole number of blocks.
    #[must_use]
    pub const fn matrix_bytes(self, rows: u64, columns: u64) -> u64 {
        rows * (columns / self.block_values()) * self.block_bytes()
    }

    /// Position in per-encoding pipeline tables.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Q8_0 => 0,
            Self::Q4K => 1,
            Self::Q6K => 2,
        }
    }
}

/// A layer's matrices in the order their encodings pack into `DecoderLayerOffsets::encodings`, four bits each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    Query = 0,
    Key = 1,
    Value = 2,
    Output = 3,
    Gate = 4,
    Up = 5,
    Down = 6,
}

/// The encoding of `matrix` in a packed field.
///
/// # Errors
///
/// Returns [`GpuError::Unavailable`] for an unimplemented encoding.
pub fn unpack(encodings: u64, matrix: Matrix) -> Result<Encoding, GpuError> {
    let id = (encodings >> (4 * matrix as u64)) & 0xF;
    Encoding::from_id(u32::try_from(id).unwrap_or(u32::MAX))
}
