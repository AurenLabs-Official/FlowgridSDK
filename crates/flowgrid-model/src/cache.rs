use burn::tensor::{backend::Backend, Tensor};

/// Per-layer KV cache used by autoregressive decoding (pre-sized ring buffer, no per-step `cat`).
#[derive(Debug)]
pub struct KvCache<B: Backend> {
    k_buf: Option<Tensor<B, 4>>,
    v_buf: Option<Tensor<B, 4>>,
    len: usize,
    max_seq: usize,
}

impl<B: Backend> KvCache<B> {
    /// Pre-allocated cache for keys/values shaped `[batch, n_kv_head, max_seq, head_dim]`.
    pub fn with_capacity(
        batch: usize,
        n_kv_head: usize,
        max_seq: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            k_buf: Some(Tensor::zeros([batch, n_kv_head, max_seq, head_dim], device)),
            v_buf: Some(Tensor::zeros([batch, n_kv_head, max_seq, head_dim], device)),
            len: 0,
            max_seq,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Append projected key/value slices `[batch, heads, seq_q, head_dim]`.
    ///
    /// Requires `len + seq_q <= max_seq` (model `block_size`).
    pub fn append(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) {
        let [b, h, seq, d] = k.dims();
        assert_eq!(v.dims(), [b, h, seq, d]);
        let (Some(kb), Some(vb)) = (&self.k_buf, &self.v_buf) else {
            panic!("KvCache::append requires KvCache::with_capacity");
        };
        assert_eq!(
            kb.dims(),
            [b, h, self.max_seq, d],
            "KV cache batch/heads/dim must match layer projections"
        );
        assert!(
            self.len + seq <= self.max_seq,
            "KV sequence overflow: len {} + {} > max_seq {}",
            self.len,
            seq,
            self.max_seq
        );
        let r = self.len..(self.len + seq);
        let k_new = kb.clone().slice_assign([0..b, 0..h, r.clone(), 0..d], k);
        let v_new = vb.clone().slice_assign([0..b, 0..h, r, 0..d], v);
        self.k_buf = Some(k_new);
        self.v_buf = Some(v_new);
        self.len += seq;
    }

    pub fn view(&self) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        let kb = self.k_buf.as_ref()?;
        let vb = self.v_buf.as_ref()?;
        let [b, h, cap, d] = kb.dims();
        if cap != self.max_seq || self.len == 0 {
            return None;
        }
        let r = 0..self.len;
        Some((
            kb.clone().slice([0..b, 0..h, r.clone(), 0..d]),
            vb.clone().slice([0..b, 0..h, r, 0..d]),
        ))
    }
}

/// Stack of per-layer KV caches.
pub type KvCacheStack<B> = Vec<KvCache<B>>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type BB = NdArray<f32>;

    #[test]
    fn prealloc_append_reaches_full_span_without_panic() {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let max_seq = 8_usize;
        let mut c: KvCache<BB> = KvCache::with_capacity(1, 2, max_seq, 4, &device);
        for step in 0..max_seq {
            let k = Tensor::<BB, 4>::zeros([1, 2, 1, 4], &device);
            let v = Tensor::<BB, 4>::zeros([1, 2, 1, 4], &device);
            c.append(k, v);
            assert_eq!(c.len(), step + 1);
        }
        let (kv, vv) = c.view().expect("view");
        assert_eq!(kv.dims(), [1, 2, max_seq, 4]);
        assert_eq!(vv.dims(), [1, 2, max_seq, 4]);
    }
}
