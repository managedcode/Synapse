use std::fmt::{Display, Formatter};

use crate::graph_execution::{OperatorError, argmax};
use crate::session_state::KvCacheOwner;

use super::tiny_dense::{TinyDenseError, TinyDenseModel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    EndOfSequence,
    MaximumTokens,
    ContextLimit,
}

impl FinishReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EndOfSequence => "end_of_sequence",
            Self::MaximumTokens => "maximum_tokens",
            Self::ContextLimit => "context_limit",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationOutput {
    pub tokens: Vec<usize>,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextGenerationError {
    EmptyPrompt,
    ZeroMaximumTokens,
    Model(TinyDenseError),
    Sampling(OperatorError),
}

impl Display for TextGenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => formatter.write_str("prompt must contain at least one token"),
            Self::ZeroMaximumTokens => formatter.write_str("maximum new tokens must be positive"),
            Self::Model(error) => write!(formatter, "tiny model failed: {error}"),
            Self::Sampling(error) => write!(formatter, "sampling failed: {error}"),
        }
    }
}

impl std::error::Error for TextGenerationError {}

impl From<TinyDenseError> for TextGenerationError {
    fn from(value: TinyDenseError) -> Self {
        Self::Model(value)
    }
}

/// Generates tokens greedily from the deterministic tiny model.
///
/// # Errors
///
/// Returns a typed input, model, or sampling error when generation cannot
/// preserve its invariants.
pub fn generate(
    model: &TinyDenseModel,
    prompt: &[usize],
    maximum_new_tokens: usize,
) -> Result<GenerationOutput, TextGenerationError> {
    if prompt.is_empty() {
        return Err(TextGenerationError::EmptyPrompt);
    }
    if maximum_new_tokens == 0 {
        return Err(TextGenerationError::ZeroMaximumTokens);
    }

    let config = model.config();
    let mut session = model.start_session(KvCacheOwner(1))?;
    let mut logits = session.prefill(prompt)?;
    let mut tokens = Vec::with_capacity(maximum_new_tokens);

    while tokens.len() < maximum_new_tokens {
        let token = argmax(&logits).map_err(TextGenerationError::Sampling)?;
        if token == config.eos_token {
            return Ok(GenerationOutput {
                tokens,
                finish_reason: FinishReason::EndOfSequence,
                prompt_tokens: prompt.len(),
            });
        }
        tokens.push(token);
        if session.position() == config.maximum_context {
            return Ok(GenerationOutput {
                tokens,
                finish_reason: FinishReason::ContextLimit,
                prompt_tokens: prompt.len(),
            });
        }
        logits = session.forward(token)?;
    }

    Ok(GenerationOutput {
        tokens,
        finish_reason: FinishReason::MaximumTokens,
        prompt_tokens: prompt.len(),
    })
}
