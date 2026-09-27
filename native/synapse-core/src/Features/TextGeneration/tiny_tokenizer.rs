use super::tiny_dense_weights::TINY_DENSE_CONFIG;

const ALPHABET: &[u8; 62] = b" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678";
const FIRST_TEXT_TOKEN: usize = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct TinyTokenizer;

impl TinyTokenizer {
    #[must_use]
    pub fn encode(text: &str) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(text.len() + 1);
        tokens.push(TINY_DENSE_CONFIG.bos_token);
        tokens.extend(text.bytes().map(|byte| {
            ALPHABET
                .iter()
                .position(|candidate| *candidate == byte)
                .map_or(FIRST_TEXT_TOKEN, |index| index + FIRST_TEXT_TOKEN)
        }));
        tokens
    }

    #[must_use]
    pub fn decode(tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|token| token.checked_sub(FIRST_TEXT_TOKEN))
            .filter_map(|index| ALPHABET.get(index))
            .map(|byte| char::from(*byte))
            .collect()
    }
}
