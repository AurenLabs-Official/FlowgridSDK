//! Memory-mapped token corpora (little-endian `u32`) and sliding-window batches.
#![allow(missing_docs)]

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::FgDataError::{Align, Empty};

#[derive(Debug, thiserror::Error)]
pub enum FgDataError {
    #[error("io: {0}")]
    Io(String),
    #[error("token storage must contain full u32 words")]
    Align,
    #[error("empty corpus")]
    Empty,
}

impl From<std::io::Error> for FgDataError {
    fn from(e: std::io::Error) -> Self {
        FgDataError::Io(e.to_string())
    }
}

pub type FgDataResult<T> = Result<T, FgDataError>;

/// Read-only mmap of `u32` token ids (little-endian raw bytes).
#[derive(Clone)]
pub struct TokenMmap {
    mmap: Arc<Mmap>,
    len_tokens: usize,
}

impl TokenMmap {
    pub fn open(path: impl AsRef<Path>) -> FgDataResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() % 4 != 0 {
            return Err(Align);
        }
        let len_tokens = mmap.len() / 4;
        if len_tokens == 0 {
            return Err(Empty);
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            len_tokens,
        })
    }

    #[inline]
    pub fn len_tokens(&self) -> usize {
        self.len_tokens
    }

    /// Raw byte view of the mmap (length is `4 * len_tokens`).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    #[inline]
    pub fn token(&self, idx: usize) -> Option<u32> {
        if idx >= self.len_tokens {
            return None;
        }
        let off = idx * 4;
        let raw = self.mmap[off..off + 4].try_into().unwrap();
        Some(u32::from_le_bytes(raw))
    }

    /// Owned raw windows (`seq_len` × `u32` LE words) for streaming loaders.
    pub fn iter_sequence_chunks(&self, seq_len: usize) -> impl Iterator<Item = Vec<u8>> + '_ {
        let mmap = self.mmap.clone();
        let bytes_per_seq = seq_len.saturating_mul(4);
        let max_start = self.len_tokens.saturating_sub(seq_len);
        (0..=max_start).map(move |start| {
            let byte_off = start * 4;
            mmap[byte_off..byte_off + bytes_per_seq].to_vec()
        })
    }
}

/// Pack raw bytes that represent `seq_len` contiguous `u32` LE ids into owned `Vec<u32>`.
pub fn sequence_bytes_to_ids(bytes: &[u8]) -> FgDataResult<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(Align);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// Write token ids as raw LE `u32` blob (for building mmap corpora).
pub fn write_token_blob(path: impl AsRef<Path>, ids: &[u32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for id in ids {
        f.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}
