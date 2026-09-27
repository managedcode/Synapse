use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatorError {
    DimensionMismatch,
    EmptyReduction,
    InvalidHeadDimension,
    NonFiniteInput,
}

impl Display for OperatorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::DimensionMismatch => "operator dimensions do not match",
            Self::EmptyReduction => "reduction input must not be empty",
            Self::InvalidHeadDimension => "RoPE head dimension must be positive and even",
            Self::NonFiniteInput => "operator input contains no finite value",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for OperatorError {}

/// Applies a row-major dense projection.
///
/// # Errors
///
/// Returns [`OperatorError::DimensionMismatch`] when the buffers do not match
/// the declared matrix dimensions.
pub fn linear(
    weights: &[f32],
    output_size: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), OperatorError> {
    if output.len() != output_size || weights.len() != output_size.saturating_mul(input.len()) {
        return Err(OperatorError::DimensionMismatch);
    }

    for (row, destination) in weights.chunks_exact(input.len()).zip(output.iter_mut()) {
        *destination = dot(row, input);
    }
    Ok(())
}

/// Applies RMS normalization.
///
/// # Errors
///
/// Returns [`OperatorError::DimensionMismatch`] for empty or mismatched
/// buffers.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn rms_norm(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), OperatorError> {
    if input.is_empty() || input.len() != weights.len() || input.len() != output.len() {
        return Err(OperatorError::DimensionMismatch);
    }

    let mean_square = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / input.len() as f64;
    let inverse_root = (mean_square + f64::from(epsilon)).sqrt().recip() as f32;

    for ((destination, value), weight) in output.iter_mut().zip(input).zip(weights) {
        *destination = *value * inverse_root * *weight;
    }
    Ok(())
}

/// Applies rotary position embeddings in place.
///
/// # Errors
///
/// Returns [`OperatorError::InvalidHeadDimension`] when the supplied head
/// shape is invalid for the input buffer.
#[allow(clippy::cast_precision_loss)]
pub fn rope(
    values: &mut [f32],
    head_count: usize,
    head_dimension: usize,
    position: usize,
    theta: f32,
) -> Result<(), OperatorError> {
    if head_dimension == 0
        || !head_dimension.is_multiple_of(2)
        || values.len() != head_count.saturating_mul(head_dimension)
    {
        return Err(OperatorError::InvalidHeadDimension);
    }

    for head in values.chunks_exact_mut(head_dimension) {
        for pair in 0..head_dimension / 2 {
            let left_index = pair * 2;
            let right_index = left_index + 1;
            let exponent = (left_index as f32) / (head_dimension as f32);
            let angle = (position as f32) / theta.powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            let left = head[left_index];
            let right = head[right_index];
            head[left_index] = left.mul_add(cosine, -right * sine);
            head[right_index] = left.mul_add(sine, right * cosine);
        }
    }
    Ok(())
}

/// Applies numerically stable softmax in place.
///
/// # Errors
///
/// Returns an error for empty input or when no finite probability mass can be
/// produced.
#[allow(clippy::cast_possible_truncation)]
pub fn softmax_in_place(values: &mut [f32]) -> Result<(), OperatorError> {
    let maximum = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f32::max)
        .ok_or(OperatorError::EmptyReduction)?;

    let mut sum = 0.0_f64;
    for value in values.iter_mut() {
        if value.is_finite() {
            *value = (*value - maximum).exp();
            sum += f64::from(*value);
        } else {
            *value = 0.0;
        }
    }

    if !sum.is_finite() || sum == 0.0 {
        return Err(OperatorError::NonFiniteInput);
    }

    let inverse_sum = sum.recip() as f32;
    for value in values.iter_mut() {
        *value *= inverse_sum;
    }
    Ok(())
}

/// Adds the source vector to the destination vector in place.
///
/// # Errors
///
/// Returns [`OperatorError::DimensionMismatch`] when lengths differ.
pub fn add_in_place(destination: &mut [f32], source: &[f32]) -> Result<(), OperatorError> {
    if destination.len() != source.len() {
        return Err(OperatorError::DimensionMismatch);
    }
    for (left, right) in destination.iter_mut().zip(source) {
        *left += right;
    }
    Ok(())
}

#[must_use]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Finds the index of the greatest finite value.
///
/// # Errors
///
/// Returns [`OperatorError::NonFiniteInput`] when no finite value exists.
pub fn argmax(values: &[f32]) -> Result<usize, OperatorError> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or(OperatorError::NonFiniteInput)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (left, right)| left.mul_add(*right, sum))
}
