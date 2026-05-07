//! Load Hugging Face **`tokenizer.json`** (and compatible SentencePiece paths).
#![allow(missing_docs)]

use std::path::Path;

/// Owned tokenizer wrapper.
pub struct FgTokenizer {
    inner: tokenizers::Tokenizer,
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

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}
