//! Load Hugging Face **`tokenizer.json`** (and compatible SentencePiece paths).
#![allow(missing_docs)]

use std::path::Path;

/// Owned tokenizer wrapper.
pub struct FgTokenizer {
    inner: tokenizers::Tokenizer,
}

#[derive(Default)]
pub struct DecoderState {
    ids: Vec<u32>,
}

impl FgTokenizer {
    /// Load from `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, tokenizers::Error> {
        let inner = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, tokenizers::Error> {
        let enc = self.inner.encode(text, add_special_tokens)?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, tokenizers::Error> {
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

    /// Decode a stream by re-decoding accumulated ids and returning the delta text.
    pub fn decode_streaming(
        &self,
        state: &mut DecoderState,
        new_id: u32,
    ) -> Result<String, tokenizers::Error> {
        let old = self.decode(&state.ids, true)?;
        state.ids.push(new_id);
        let now = self.decode(&state.ids, true)?;
        Ok(now.strip_prefix(&old).unwrap_or(&now).to_string())
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
