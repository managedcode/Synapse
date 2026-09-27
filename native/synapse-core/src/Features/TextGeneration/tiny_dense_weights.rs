pub const TINY_DENSE_CONFIG: TinyDenseConfig = TinyDenseConfig {
    block_count: 2,
    hidden_size: 32,
    feed_forward_size: 64,
    query_head_count: 4,
    kv_head_count: 2,
    head_dimension: 8,
    vocabulary_size: 64,
    maximum_context: 128,
    rms_epsilon: 1.0e-5,
    rope_theta: 10_000.0,
    eos_token: 0,
    bos_token: 1,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TinyDenseConfig {
    pub block_count: usize,
    pub hidden_size: usize,
    pub feed_forward_size: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub head_dimension: usize,
    pub vocabulary_size: usize,
    pub maximum_context: usize,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub eos_token: usize,
    pub bos_token: usize,
}

#[derive(Debug)]
pub(super) struct TinyDenseWeights {
    pub embeddings: Vec<f32>,
    pub blocks: Vec<TinyDenseBlockWeights>,
    pub final_norm: Vec<f32>,
    fingerprint: u64,
}

#[derive(Debug)]
pub(super) struct TinyDenseBlockWeights {
    pub attention_norm: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub feed_forward_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

impl TinyDenseWeights {
    pub fn generate(config: TinyDenseConfig, seed: u64) -> Self {
        let mut generator = WeightGenerator::new(seed);
        let query_size = config.query_head_count * config.head_dimension;
        let kv_size = config.kv_head_count * config.head_dimension;
        let mut weights = Self {
            embeddings: generator.values(config.vocabulary_size * config.hidden_size, 0.11),
            blocks: (0..config.block_count)
                .map(|_| TinyDenseBlockWeights {
                    attention_norm: generator.norm_values(config.hidden_size),
                    query: generator.values(query_size * config.hidden_size, 0.09),
                    key: generator.values(kv_size * config.hidden_size, 0.09),
                    value: generator.values(kv_size * config.hidden_size, 0.09),
                    attention_output: generator.values(config.hidden_size * query_size, 0.09),
                    feed_forward_norm: generator.norm_values(config.hidden_size),
                    gate: generator.values(config.feed_forward_size * config.hidden_size, 0.08),
                    up: generator.values(config.feed_forward_size * config.hidden_size, 0.08),
                    down: generator.values(config.hidden_size * config.feed_forward_size, 0.08),
                })
                .collect(),
            final_norm: generator.norm_values(config.hidden_size),
            fingerprint: 0,
        };
        weights.fingerprint = weights.calculate_fingerprint();
        weights
    }

    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn calculate_fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.add_slice(&self.embeddings);
        for block in &self.blocks {
            hash.add_slice(&block.attention_norm);
            hash.add_slice(&block.query);
            hash.add_slice(&block.key);
            hash.add_slice(&block.value);
            hash.add_slice(&block.attention_output);
            hash.add_slice(&block.feed_forward_norm);
            hash.add_slice(&block.gate);
            hash.add_slice(&block.up);
            hash.add_slice(&block.down);
        }
        hash.add_slice(&self.final_norm);
        hash.value
    }
}

struct WeightGenerator {
    state: u64,
}

impl WeightGenerator {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_value(scale)).collect()
    }

    fn norm_values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| 1.0 + self.next_value(0.025)).collect()
    }

    #[allow(clippy::cast_precision_loss)]
    fn next_value(&mut self, scale: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = ((self.state >> 40) & 0x00ff_ffff) as u32;
        let unit = (mantissa as f32) / 16_777_216.0;
        (unit * 2.0 - 1.0) * scale
    }
}

struct Fingerprint {
    value: u64,
}

impl Fingerprint {
    const fn new() -> Self {
        Self {
            value: 0xcbf2_9ce4_8422_2325,
        }
    }

    fn add_slice(&mut self, values: &[f32]) {
        for value in values {
            for byte in value.to_bits().to_le_bytes() {
                self.value ^= u64::from(byte);
                self.value = self.value.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
}
