use burn::tensor::{backend::Backend, Tensor};

/// Per-layer KV cache used by autoregressive decoding.
#[derive(Debug)]
pub struct KvCache<B: Backend> {
    k: Option<Tensor<B, 3>>,
    v: Option<Tensor<B, 3>>,
}

impl<B: Backend> KvCache<B> {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    /// Append/replace cache state with projected key/value tensors.
    pub fn append(&mut self, k: Tensor<B, 3>, v: Tensor<B, 3>) {
        self.k = Some(k);
        self.v = Some(v);
    }

    pub fn view(&self) -> Option<(Tensor<B, 3>, Tensor<B, 3>)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }
}

/// Stack of per-layer KV caches.
pub type KvCacheStack<B> = Vec<KvCache<B>>;
