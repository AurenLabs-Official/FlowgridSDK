//! Deterministic greedy / stochastic sampling from last-timestep logits.

use burn::tensor::{backend::Backend, Tensor};
use num_traits::ToPrimitive;
use rand::Rng;

/// How to pick the next token from the final position logits `[1, seq, vocab]`.
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Greedy,
    Temperature { t: f32 },
    TopK { k: usize, t: f32 },
}

fn softmax_temp(logits: &mut [f32], t: f32) {
    let t = t.max(1e-8);
    let mut max = f32::NEG_INFINITY;
    for &x in logits.iter() {
        if x.is_finite() && x > max {
            max = x;
        }
    }
    let mut s = 0.0f32;
    for x in logits.iter_mut() {
        let e = ((*x - max) / t).exp();
        *x = e;
        s += e;
    }
    if s > 0.0 {
        for x in logits.iter_mut() {
            *x /= s;
        }
    }
}

/// Sample next token id from logits of shape `[batch, seq, vocab]` using the last time step.
pub fn sample_from_last_logits<B: Backend, R: Rng + ?Sized>(
    logits: &Tensor<B, 3>,
    sampling: Sampling,
    rng: &mut R,
) -> i32 {
    let [_, seq, vocab] = logits.dims();
    let row = logits.clone().slice([0..1, (seq - 1)..seq, 0..vocab]);
    match sampling {
        Sampling::Greedy => {
            let idx = row.argmax(2);
            let s = idx.into_scalar();
            ToPrimitive::to_i32(&s).unwrap_or(0)
        }
        Sampling::Temperature { t } => {
            let mut data = row.reshape([vocab]).into_data().convert::<f32>().value;
            softmax_temp(&mut data, t);
            let idx = sample_multinomial(&data, rng);
            idx as i32
        }
        Sampling::TopK { k, t } => {
            let mut data = row.reshape([vocab]).into_data().convert::<f32>().value;
            let k = k.max(1).min(data.len());
            let mut order: Vec<usize> = (0..data.len()).collect();
            order.sort_by(|&a, &b| {
                data[b]
                    .partial_cmp(&data[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut masked = vec![f32::NEG_INFINITY; data.len()];
            for &i in order.iter().take(k) {
                masked[i] = data[i];
            }
            data = masked;
            softmax_temp(&mut data, t);
            let idx = sample_multinomial(&data, rng);
            idx as i32
        }
    }
}

fn sample_multinomial<R: Rng + ?Sized>(probs: &[f32], rng: &mut R) -> usize {
    let r: f32 = rng.gen();
    let mut c = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        c += p;
        if r <= c || i + 1 == probs.len() {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}
