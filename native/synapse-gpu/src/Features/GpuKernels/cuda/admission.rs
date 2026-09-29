//! What the CUDA kernels implement: FP32 KV, `Q8_0` matrices, and head dimension 64. Anything else fails before the
//! driver is touched, so the explicit error never depends on having an NVIDIA device.

use crate::gpu_kernels::decoder::{ALL_Q8_0, CUDA_HEAD_DIM, DecoderPlan, Encoding, KvPrecision};
use crate::gpu_kernels::error::GpuError;

/// Rejects plans the CUDA kernels do not implement.
///
/// # Errors
///
/// Returns [`GpuError::Unavailable`] for FP16 KV, K-quant weights (ADR-021), or a head dimension other than 64.
pub fn admit(plan: &DecoderPlan) -> Result<(), GpuError> {
    if plan.shape.kv_precision != KvPrecision::F32 {
        return Err(GpuError::Unavailable(
            "the FP16 KV profile is implemented for Metal only; CUDA uses FP32 KV".into(),
        ));
    }

    let all_q8 = plan.embedding_encoding == Encoding::Q8_0
        && plan.output_encoding == Encoding::Q8_0
        && plan.layers.iter().all(|layer| layer.encodings == ALL_Q8_0);
    if !all_q8 {
        return Err(GpuError::Unavailable(
            "the CUDA kernels implement Q8_0 matrices; K-quant weights run on Metal (ADR-021)"
                .into(),
        ));
    }

    if plan.shape.head_dim != CUDA_HEAD_DIM {
        return Err(GpuError::Unavailable(format!(
            "the CUDA kernels implement head dimension {CUDA_HEAD_DIM}; the model uses {} (Metal compiles 64 and 128)",
            plan.shape.head_dim
        )));
    }

    Ok(())
}
