//! Load Hugging Face **`tokenizer.json`** (and compatible SentencePiece paths).
#![allow(missing_docs)]

use std::path::Path;

/// Owned tokenizer wrapper.
pub struct FgTokenizer {
    inner: tokenizers::Tokenizer,
}

/// State for [`FgTokenizer::decode_streaming`]; holds accumulated ids and last full decode.
#[derive(Default)]
pub struct DecoderState {
    ids: Vec<u32>,
    prev_decoded_full: String,
}

impl DecoderState {
    pub fn reset(&mut self) {
        self.ids.clear();
        self.prev_decoded_full.clear();
    }
}

impl FgTokenizer {
    /// Load from `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, tokenizers::Error> {
        let inner = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, tokenizers::Error> {
        let enc = self.inner.encode(text, add_special_tokens)?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, tokenizers::Error> {
        self.inner.decode(ids, skip_special_tokens)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.inner.token_to_id("<bos>")
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.inner.token_to_id("<eos>")
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.inner.token_to_id("<pad>")
    }

    /// Decode a stream incrementally: **one** full `decode` per new token, delta via prefix / LCP vs
    /// cached full string (avoids O(n) double-decodes per token).
    ///
    /// If the tokenizer rewrites earlier text (BPE / SPM), the fallback uses the longest UTF-8-safe
    /// common prefix between the previous and current full decodes.
    pub fn decode_streaming(
        &self,
        state: &mut DecoderState,
        new_id: u32,
    ) -> Result<String, tokenizers::Error> {
        state.ids.push(new_id);
        let full = self.decode(&state.ids, true)?;

        let delta = if let Some(rest) = full.strip_prefix(state.prev_decoded_full.as_str()) {
            rest.to_string()
        } else {
            let common = state
                .prev_decoded_full
                .as_bytes()
                .iter()
                .zip(full.as_bytes())
                .take_while(|(a, b)| a == b)
                .count();
            let cut = (0..=common)
                .rev()
                .find(|i| full.is_char_boundary(*i))
                .unwrap_or(0);
            full[cut..].to_string()
        };
        state.prev_decoded_full = full;
        Ok(delta)
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
