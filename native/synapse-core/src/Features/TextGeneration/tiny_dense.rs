use std::fmt::{Display, Formatter};

use crate::graph_execution::{
    OperatorError, add_in_place, linear, rms_norm, rope, silu, softmax_in_place,
};
use crate::session_state::{KvCacheError, KvCacheOwner, PagedKvCache};

use super::tiny_dense_weights::{TINY_DENSE_CONFIG, TinyDenseConfig, TinyDenseWeights};

const FIXTURE_SEED: u64 = 0x5359_4e41_5053_4531;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TinyDenseError {
    TokenOutOfRange,
    ContextLimitExceeded,
    Operator(OperatorError),
    KvCache(KvCacheError),
}

impl Display for TinyDenseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenOutOfRange => formatter.write_str("token is outside the tiny vocabulary"),
            Self::ContextLimitExceeded => formatter.write_str("tiny model context limit exceeded"),
            Self::Operator(error) => write!(formatter, "reference operator failed: {error}"),
            Self::KvCache(error) => write!(formatter, "KV cache failed: {error}"),
        }
    }
}

impl std::error::Error for TinyDenseError {}

impl From<OperatorError> for TinyDenseError {
    fn from(value: OperatorError) -> Self {
        Self::Operator(value)
    }
}

impl From<KvCacheError> for TinyDenseError {
    fn from(value: KvCacheError) -> Self {
        Self::KvCache(value)
    }
}

#[derive(Debug)]
pub struct TinyDenseModel {
    config: TinyDenseConfig,
    weights: TinyDenseWeights,
}

impl TinyDenseModel {
    #[must_use]
    pub fn fixture() -> Self {
        Self {
            config: TINY_DENSE_CONFIG,
            weights: TinyDenseWeights::generate(TINY_DENSE_CONFIG, FIXTURE_SEED),
        }
    }

    #[must_use]
    pub const fn config(&self) -> TinyDenseConfig {
        self.config
    }

    #[must_use]
    pub const fn weights_fingerprint(&self) -> u64 {
        self.weights.fingerprint()
    }

    /// Starts an isolated bounded generation session.
    ///
    /// # Errors
    ///
    /// Returns a KV-cache configuration error when the model shape is invalid.
    pub fn start_session(
        &self,
        owner: KvCacheOwner,
    ) -> Result<TinyDenseSession<'_>, TinyDenseError> {
        let cache = PagedKvCache::new(
            owner,
            self.config.block_count,
            self.config.kv_head_count,
            self.config.head_dimension,
            self.config.maximum_context,
        )?;
        Ok(TinyDenseSession {
            model: self,
            cache,
            owner,
            position: 0,
        })
    }
}

pub struct TinyDenseSession<'model> {
    model: &'model TinyDenseModel,
    cache: PagedKvCache,
    owner: KvCacheOwner,
    position: usize,
}

impl TinyDenseSession<'_> {
    /// Executes one autoregressive token and commits its KV state atomically.
    ///
    /// # Errors
    ///
    /// Returns a typed token, context, operator, or KV-cache error. KV state is
    /// rolled back when execution fails.
    pub fn forward(&mut self, token: usize) -> Result<Vec<f32>, TinyDenseError> {
        if token >= self.model.config.vocabulary_size {
            return Err(TinyDenseError::TokenOutOfRange);
        }
        if self.position >= self.model.config.maximum_context {
            return Err(TinyDenseError::ContextLimitExceeded);
        }

        let original_position = self.position;
        match self.forward_inner(token) {
            Ok(logits) => {
                self.position += 1;
                Ok(logits)
            }
            Err(error) => {
                self.cache.rollback(self.owner, original_position)?;
                Err(error)
            }
        }
    }

    /// Executes a sequence of prompt tokens.
    ///
    /// # Errors
    ///
    /// Returns the first typed error produced by [`Self::forward`].
    pub fn prefill(&mut self, tokens: &[usize]) -> Result<Vec<f32>, TinyDenseError> {
        let mut logits = Vec::new();
        for token in tokens {
            logits = self.forward(*token)?;
        }
        Ok(logits)
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    fn forward_inner(&mut self, token: usize) -> Result<Vec<f32>, TinyDenseError> {
        let config = self.model.config;
        let embedding_start = token * config.hidden_size;
        let embedding_end = embedding_start + config.hidden_size;
        let mut hidden = self.model.weights.embeddings[embedding_start..embedding_end].to_vec();

        for (layer_index, block) in self.model.weights.blocks.iter().enumerate() {
            let mut normalized = vec![0.0; config.hidden_size];
            rms_norm(
                &hidden,
                &block.attention_norm,
                config.rms_epsilon,
                &mut normalized,
            )?;

            let query_size = config.query_head_count * config.head_dimension;
            let kv_size = config.kv_head_count * config.head_dimension;
            let mut query = vec![0.0; query_size];
            let mut key = vec![0.0; kv_size];
            let mut value = vec![0.0; kv_size];
            linear(&block.query, query_size, &normalized, &mut query)?;
            linear(&block.key, kv_size, &normalized, &mut key)?;
            linear(&block.value, kv_size, &normalized, &mut value)?;
            rope(
                &mut query,
                config.query_head_count,
                config.head_dimension,
                self.position,
                config.rope_theta,
            )?;
            rope(
                &mut key,
                config.kv_head_count,
                config.head_dimension,
                self.position,
                config.rope_theta,
            )?;
            self.cache
                .append(self.owner, layer_index, self.position, &key, &value)?;

            let attention = self.attention(layer_index, &query)?;
            let mut projected_attention = vec![0.0; config.hidden_size];
            linear(
                &block.attention_output,
                config.hidden_size,
                &attention,
                &mut projected_attention,
            )?;
            add_in_place(&mut hidden, &projected_attention)?;

            rms_norm(
                &hidden,
                &block.feed_forward_norm,
                config.rms_epsilon,
                &mut normalized,
            )?;
            let mut gate = vec![0.0; config.feed_forward_size];
            let mut up = vec![0.0; config.feed_forward_size];
            linear(
                &block.gate,
                config.feed_forward_size,
                &normalized,
                &mut gate,
            )?;
            linear(&block.up, config.feed_forward_size, &normalized, &mut up)?;
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value = silu(*gate_value) * up_value;
            }
            let mut feed_forward = vec![0.0; config.hidden_size];
            linear(&block.down, config.hidden_size, &gate, &mut feed_forward)?;
            add_in_place(&mut hidden, &feed_forward)?;
        }

        let mut normalized = vec![0.0; config.hidden_size];
        rms_norm(
            &hidden,
            &self.model.weights.final_norm,
            config.rms_epsilon,
            &mut normalized,
        )?;
        let mut logits = vec![0.0; config.vocabulary_size];
        linear(
            &self.model.weights.embeddings,
            config.vocabulary_size,
            &normalized,
            &mut logits,
        )?;
        Ok(logits)
    }

    #[allow(clippy::cast_precision_loss)]
    fn attention(&self, layer: usize, query: &[f32]) -> Result<Vec<f32>, TinyDenseError> {
        let config = self.model.config;
        let mut output = vec![0.0; config.query_head_count * config.head_dimension];
        let group_size = config.query_head_count / config.kv_head_count;
        let scale = (config.head_dimension as f32).sqrt().recip();

        for query_head in 0..config.query_head_count {
            let kv_head = query_head / group_size;
            let head_start = query_head * config.head_dimension;
            let head_end = head_start + config.head_dimension;
            let query_values = &query[head_start..head_end];
            let mut scores = Vec::with_capacity(self.position + 1);
            for key_position in 0..=self.position {
                let key = self
                    .cache
                    .key(layer, key_position, kv_head, config.head_dimension)?;
                scores.push(dot(query_values, key) * scale);
            }
            softmax_in_place(&mut scores)?;

            for (key_position, probability) in scores.into_iter().enumerate() {
                let value =
                    self.cache
                        .value(layer, key_position, kv_head, config.head_dimension)?;
                for (destination, source) in output[head_start..head_end].iter_mut().zip(value) {
                    *destination = source.mul_add(probability, *destination);
                }
            }
        }
        Ok(output)
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (left, right)| left.mul_add(*right, sum))
}
