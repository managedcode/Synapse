//! Page-aligned weight segments (ADR-019): the byte ranges a plan reads. A dropped layer's tensors lie in no
//! segment, so a backend that maps segments never makes their pages resident.

use super::{DecoderLayerOffsets, DecoderPlan, DecoderShape, Encoding, Matrix, NO_TENSOR};

/// Bytes of a matrix of a validated layer; an unreadable encoding (never true after validation) counts as `Q8_0`.
fn bytes(layer: &DecoderLayerOffsets, matrix: Matrix, rows: u64, columns: u64) -> u64 {
    layer
        .encoding(matrix)
        .unwrap_or(Encoding::Q8_0)
        .matrix_bytes(rows, columns)
}

/// `[first byte, end)` of every tensor of one layer.
fn layer_extent(shape: &DecoderShape, layer: &DecoderLayerOffsets) -> (u64, u64) {
    let (hidden, kv, ffn) = (
        u64::from(shape.hidden),
        u64::from(shape.kv_width()),
        u64::from(shape.feed_forward),
    );
    [
        (layer.attention_norm, hidden * 4),
        (layer.query, bytes(layer, Matrix::Query, hidden, hidden)),
        (layer.key, bytes(layer, Matrix::Key, kv, hidden)),
        (layer.value, bytes(layer, Matrix::Value, kv, hidden)),
        (layer.query_bias, hidden * 4),
        (layer.key_bias, kv * 4),
        (layer.value_bias, kv * 4),
        (
            layer.attention_output,
            bytes(layer, Matrix::Output, hidden, hidden),
        ),
        (layer.feed_forward_norm, hidden * 4),
        (layer.gate, bytes(layer, Matrix::Gate, ffn, hidden)),
        (layer.up, bytes(layer, Matrix::Up, ffn, hidden)),
        (layer.down, bytes(layer, Matrix::Down, hidden, ffn)),
    ]
    .into_iter()
    .filter(|&(offset, _)| offset != NO_TENSOR)
    .fold((u64::MAX, 0), |(first, end), (offset, bytes)| {
        (first.min(offset), end.max(offset + bytes))
    })
}

impl DecoderPlan {
    /// Page-aligned `[start, end)` byte ranges covering every tensor the plan reads. Each unit (the token
    /// embedding, the output norm, the output projection, and each layer) lies inside one range, so one launch
    /// binds one segment; units whose pages touch share a range.
    #[must_use]
    pub fn weight_segments(&self, page: u64) -> Vec<(u64, u64)> {
        let (hidden, vocabulary) = (
            u64::from(self.shape.hidden),
            u64::from(self.shape.vocabulary),
        );
        let mut spans: Vec<(u64, u64)> = [
            (
                self.token_embedding,
                self.token_embedding + self.embedding_encoding.matrix_bytes(vocabulary, hidden),
            ),
            (
                self.output,
                self.output + self.output_encoding.matrix_bytes(vocabulary, hidden),
            ),
            (self.output_norm, self.output_norm + hidden * 4),
        ]
        .into_iter()
        .chain(
            self.layers
                .iter()
                .map(|layer| layer_extent(&self.shape, layer)),
        )
        .map(|(start, end)| (start - start % page, end.next_multiple_of(page)))
        .collect();
        spans.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }

        merged
    }
}
