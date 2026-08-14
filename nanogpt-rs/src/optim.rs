//! AdamW optimizer with decoupled weight decay, matching
//! `torch.optim.AdamW` semantics.

/// Per-parameter-group weight decay is handled via a per-element `decay`
/// mask on the flat parameter buffer (2-D weights decay; biases and
/// LayerNorm gains don't), mirroring nanoGPT's `configure_optimizers`.
pub struct AdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamW {
    pub fn new(n_params: usize, beta1: f32, beta2: f32, eps: f32) -> Self {
        AdamW { m: vec![0.0; n_params], v: vec![0.0; n_params], t: 0, beta1, beta2, eps }
    }

    /// One optimization step. `lr` is the current learning rate, `wd` the
    /// weight decay coefficient (applied decoupled, only where
    /// `decay[i] == true`).
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], decay: &[bool], lr: f32, wd: f32) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let (b1, b2, eps) = (self.beta1, self.beta2, self.eps);
        let decay_factor = 1.0 - lr * wd;
        for i in 0..params.len() {
            let g = grads[i];
            let p = &mut params[i];
            if decay[i] {
                *p *= decay_factor; // decoupled weight decay
            }
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            let mhat = *m / bc1;
            let vhat = *v / bc2;
            *p -= lr * mhat / (vhat.sqrt() + eps);
        }
    }
}

/// Learning-rate schedule from train.py: linear warmup, then cosine decay
/// down to `min_lr`, held constant after `lr_decay_iters`.
pub fn get_lr(it: usize, warmup_iters: usize, lr_decay_iters: usize, learning_rate: f32, min_lr: f32) -> f32 {
    if it < warmup_iters {
        return learning_rate * (it + 1) as f32 / (warmup_iters + 1) as f32;
    }
    if it > lr_decay_iters {
        return min_lr;
    }
    let decay_ratio = (it - warmup_iters) as f32 / (lr_decay_iters - warmup_iters) as f32;
    debug_assert!((0.0..=1.0).contains(&decay_ratio));
    let coeff = 0.5 * (1.0 + (std::f32::consts::PI * decay_ratio).cos());
    min_lr + coeff * (learning_rate - min_lr)
}

/// Global-norm gradient clipping; returns the total norm. Clipped in place
/// when the norm exceeds `max_norm`.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let mut sq = 0.0f64;
    for &g in grads.iter() {
        sq += (g as f64) * (g as f64);
    }
    let norm = sq.sqrt() as f32;
    if max_norm > 0.0 && norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        for g in grads.iter_mut() {
            *g *= s;
        }
    }
    norm
}
