/// Linear warmup followed by cosine decay.
pub fn lr(step: usize, total: usize, warmup: usize, base_lr: f64, min_lr: f64) -> f64 {
    if total == 0 {
        return base_lr;
    }
    if step < warmup.max(1) {
        return base_lr * (step as f64 + 1.0) / warmup.max(1) as f64;
    }
    let denom = (total.saturating_sub(warmup)).max(1) as f64;
    let progress = (step.saturating_sub(warmup)) as f64 / denom;
    let cos = (std::f64::consts::PI * progress).cos();
    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos)
}
