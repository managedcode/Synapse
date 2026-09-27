use std::fmt::{Display, Formatter};

pub const KV_PAGE_TOKENS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheOwner(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheError {
    InvalidConfiguration,
    WrongOwner,
    InvalidLayer,
    InvalidPosition,
    ContextLimitExceeded,
    DimensionMismatch,
}

impl Display for KvCacheError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidConfiguration => "KV cache configuration is invalid",
            Self::WrongOwner => "KV cache append attempted by a non-owner session",
            Self::InvalidLayer => "KV cache layer index is invalid",
            Self::InvalidPosition => "KV cache append position is not contiguous",
            Self::ContextLimitExceeded => "KV cache context limit exceeded",
            Self::DimensionMismatch => "KV cache key/value dimensions do not match",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for KvCacheError {}

#[derive(Debug)]
pub struct PagedKvCache {
    owner: KvCacheOwner,
    kv_stride: usize,
    maximum_context: usize,
    layers: Vec<LayerCache>,
}

impl PagedKvCache {
    /// Creates an empty bounded KV cache owned by one session.
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheError::InvalidConfiguration`] for zero dimensions or
    /// arithmetic overflow.
    pub fn new(
        owner: KvCacheOwner,
        layer_count: usize,
        kv_heads: usize,
        head_dimension: usize,
        maximum_context: usize,
    ) -> Result<Self, KvCacheError> {
        if layer_count == 0 || kv_heads == 0 || head_dimension == 0 || maximum_context == 0 {
            return Err(KvCacheError::InvalidConfiguration);
        }
        let kv_stride = kv_heads
            .checked_mul(head_dimension)
            .ok_or(KvCacheError::InvalidConfiguration)?;
        let layers = (0..layer_count).map(|_| LayerCache::default()).collect();
        Ok(Self {
            owner,
            kv_stride,
            maximum_context,
            layers,
        })
    }

    /// Appends one key/value token to a layer.
    ///
    /// # Errors
    ///
    /// Returns a typed ownership, shape, position, layer, or context error when
    /// the append would violate cache invariants.
    pub fn append(
        &mut self,
        owner: KvCacheOwner,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        if owner != self.owner {
            return Err(KvCacheError::WrongOwner);
        }
        if key.len() != self.kv_stride || value.len() != self.kv_stride {
            return Err(KvCacheError::DimensionMismatch);
        }
        if position >= self.maximum_context {
            return Err(KvCacheError::ContextLimitExceeded);
        }
        let layer_cache = self
            .layers
            .get_mut(layer)
            .ok_or(KvCacheError::InvalidLayer)?;
        if position != layer_cache.length {
            return Err(KvCacheError::InvalidPosition);
        }

        let page_index = position / KV_PAGE_TOKENS;
        let page_offset = position % KV_PAGE_TOKENS;
        if page_index == layer_cache.pages.len() {
            layer_cache.pages.push(KvPage::new(self.kv_stride));
        }
        let page = &mut layer_cache.pages[page_index];
        let start = page_offset * self.kv_stride;
        let end = start + self.kv_stride;
        page.keys[start..end].copy_from_slice(key);
        page.values[start..end].copy_from_slice(value);
        page.valid_tokens = page.valid_tokens.max(page_offset + 1);
        layer_cache.length += 1;
        Ok(())
    }

    /// Reads one cached key head.
    ///
    /// # Errors
    ///
    /// Returns a typed layer, position, or dimension error for invalid access.
    pub fn key(
        &self,
        layer: usize,
        position: usize,
        kv_head: usize,
        head_dimension: usize,
    ) -> Result<&[f32], KvCacheError> {
        self.read(layer, position, kv_head, head_dimension, true)
    }

    /// Reads one cached value head.
    ///
    /// # Errors
    ///
    /// Returns a typed layer, position, or dimension error for invalid access.
    pub fn value(
        &self,
        layer: usize,
        position: usize,
        kv_head: usize,
        head_dimension: usize,
    ) -> Result<&[f32], KvCacheError> {
        self.read(layer, position, kv_head, head_dimension, false)
    }

    #[must_use]
    pub fn layer_length(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(|cache| cache.length)
    }

    /// Truncates every layer to the same committed token length.
    ///
    /// # Errors
    ///
    /// Returns [`KvCacheError::WrongOwner`] for another session or
    /// [`KvCacheError::InvalidPosition`] when the target is ahead of a layer.
    pub fn rollback(&mut self, owner: KvCacheOwner, new_length: usize) -> Result<(), KvCacheError> {
        if owner != self.owner {
            return Err(KvCacheError::WrongOwner);
        }
        for layer in &mut self.layers {
            if new_length > layer.length {
                return Err(KvCacheError::InvalidPosition);
            }
            layer.length = new_length;
            let required_pages = new_length.div_ceil(KV_PAGE_TOKENS);
            layer.pages.truncate(required_pages);
            if let Some(last_page) = layer.pages.last_mut() {
                last_page.valid_tokens = new_length % KV_PAGE_TOKENS;
                if last_page.valid_tokens == 0 && new_length > 0 {
                    last_page.valid_tokens = KV_PAGE_TOKENS;
                }
            }
        }
        Ok(())
    }

    fn read(
        &self,
        layer: usize,
        position: usize,
        kv_head: usize,
        head_dimension: usize,
        read_key: bool,
    ) -> Result<&[f32], KvCacheError> {
        if head_dimension == 0 || !self.kv_stride.is_multiple_of(head_dimension) {
            return Err(KvCacheError::DimensionMismatch);
        }
        let layer_cache = self.layers.get(layer).ok_or(KvCacheError::InvalidLayer)?;
        if position >= layer_cache.length {
            return Err(KvCacheError::InvalidPosition);
        }
        let head_count = self.kv_stride / head_dimension;
        if kv_head >= head_count {
            return Err(KvCacheError::DimensionMismatch);
        }
        let page = &layer_cache.pages[position / KV_PAGE_TOKENS];
        let page_offset = position % KV_PAGE_TOKENS;
        if page_offset >= page.valid_tokens {
            return Err(KvCacheError::InvalidPosition);
        }
        let start = page_offset * self.kv_stride + kv_head * head_dimension;
        let end = start + head_dimension;
        let data = if read_key { &page.keys } else { &page.values };
        Ok(&data[start..end])
    }
}

#[derive(Debug, Default)]
struct LayerCache {
    pages: Vec<KvPage>,
    length: usize,
}

#[derive(Debug)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
    valid_tokens: usize,
}

impl KvPage {
    fn new(kv_stride: usize) -> Self {
        let element_count = KV_PAGE_TOKENS * kv_stride;
        Self {
            keys: vec![0.0; element_count],
            values: vec![0.0; element_count],
            valid_tokens: 0,
        }
    }
}
