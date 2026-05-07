use burn::optim::GradientsParams;

/// Placeholder for gradient clipping integration.
///
/// Burn 0.13 exposes optimizer internals that vary by backend; the function is
/// intentionally no-op in this phase while keeping a stable call-site contract.
pub fn clip_grad_norm(
    _grads: &mut GradientsParams,
    _max_norm: f32,
) {
}
