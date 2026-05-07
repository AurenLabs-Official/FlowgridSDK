use burn::optim::GradientsParams;
use burn::grad_clipping::GradientClippingConfig;

/// Placeholder for gradient clipping integration.
///
/// Burn 0.13 exposes optimizer internals that vary by backend; the function is
/// intentionally no-op in this phase while keeping a stable call-site contract.
pub fn clip_grad_norm(
    _grads: &mut GradientsParams,
    _max_norm: f32,
) {
}

pub fn grad_clip_config(max_norm: Option<f32>) -> Option<GradientClippingConfig> {
    max_norm.map(GradientClippingConfig::Norm)
}
