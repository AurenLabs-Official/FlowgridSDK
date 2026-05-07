//! ROME-style rank-one updates on an FFN layer (scaffolding).

use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{FgError, FgResult};

/// Apply `W <- W + u v^T` on a rank-2 weight tensor (host-side toy implementation).
///
/// Production path operates module params via Burn records; this proves the linear algebra shape.
pub fn rank_one_update_ffn<B: Backend>(
    mut weight: Tensor<B, 2>,
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
) -> FgResult<Tensor<B, 2>> {
    let outer = u.matmul(v.transpose());
    if outer.shape() != weight.shape() {
        return Err(FgError::shape("ROME outer product shape mismatch"));
    }
    weight = weight + outer;
    Ok(weight)
}
