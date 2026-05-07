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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, Copy)]
pub struct SplitSpec {
    /// Train fraction in `[0.0, 1.0]`.
    pub train_frac: f32,
    /// Validation fraction in `[0.0, 1.0]`.
    pub val_frac: f32,
}

impl SplitSpec {
    pub fn normalized(train_frac: f32, val_frac: f32) -> Self {
        let train = train_frac.clamp(0.0, 1.0);
        let val = val_frac.clamp(0.0, 1.0);
        let sum = train + val;
        if sum >= 0.999_999 {
            // Keep at least a tiny non-train/val remainder for test if possible.
            let scale = 0.999_999 / sum.max(1e-6);
            return Self {
                train_frac: train * scale,
                val_frac: val * scale,
            };
        }
        Self {
            train_frac: train,
            val_frac: val,
        }
    }
}

/// Compute `[start, end)` token bounds for a dataset split.
pub fn split_bounds(total_tokens: usize, spec: SplitSpec, split: DatasetSplit) -> (usize, usize) {
    let train_end = ((total_tokens as f64) * spec.train_frac as f64).round() as usize;
    let val_end = ((total_tokens as f64) * (spec.train_frac + spec.val_frac) as f64).round() as usize;
    let train_end = train_end.min(total_tokens);
    let val_end = val_end.min(total_tokens).max(train_end);
    match split {
        DatasetSplit::Train => (0, train_end),
        DatasetSplit::Val => (train_end, val_end),
        DatasetSplit::Test => (val_end, total_tokens),
    }
}

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

    /// Bounds `[start, end)` for a logical split over the mmap.
    pub fn split_bounds(&self, spec: SplitSpec, split: DatasetSplit) -> (usize, usize) {
        split_bounds(self.len_tokens, spec, split)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_bounds_partition_full_range() {
        let spec = SplitSpec::normalized(0.8, 0.1);
        let (a0, a1) = split_bounds(100, spec, DatasetSplit::Train);
        let (b0, b1) = split_bounds(100, spec, DatasetSplit::Val);
        let (c0, c1) = split_bounds(100, spec, DatasetSplit::Test);
        assert_eq!((a0, a1), (0, 80));
        assert_eq!((b0, b1), (80, 90));
        assert_eq!((c0, c1), (90, 100));
        assert_eq!(a1, b0);
        assert_eq!(b1, c0);
    }
}
