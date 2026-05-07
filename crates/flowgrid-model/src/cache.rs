use burn::tensor::{backend::Backend, Tensor};

/// Per-layer KV cache used by autoregressive decoding.
#[derive(Debug)]
pub struct KvCache<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    /// Append/replace cache state with projected key/value tensors.
    pub fn append(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) {
        match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                self.k = Some(Tensor::cat(vec![prev_k.clone(), k], 2));
                self.v = Some(Tensor::cat(vec![prev_v.clone(), v], 2));
            }
            _ => {
                self.k = Some(k);
                self.v = Some(v);
            }
        }
    }

    pub fn view(&self) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }
}

/// Stack of per-layer KV caches.
pub type KvCacheStack<B> = Vec<KvCache<B>>;
