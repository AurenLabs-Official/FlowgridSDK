use burn::tensor::{backend::Backend, Int, Tensor};

use crate::cache::KvCacheStack;

/// Common trait used by CLI/serve/eval for decoder-only models.
pub trait LmModel<B: Backend> {
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
    fn forward_step(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: Option<&mut KvCacheStack<B>>,
    ) -> Tensor<B, 3>;
    fn block_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn n_layer(&self) -> usize;
}
