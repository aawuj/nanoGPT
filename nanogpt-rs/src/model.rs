//! GPT model definition — Rust port of nanoGPT's `model.py`.
//!
//! Faithful translation of the PyTorch version:
//! * token + positional embeddings, optional dropout
//! * pre-LayerNorm transformer blocks: causal multi-head self-attention
//!   (one fused QKV projection) + MLP with GELU (tanh approximation, the
//!   same variant GPT-2 uses)
//! * weight tying between `wte` and the lm-head (`lm_head.weight = wte`)
//! * GPT-2 style init: N(0, 0.02), residual projections scaled by
//!   1/sqrt(2 * n_layer)
//!
//! All parameters live in one flat `Vec<f32>`; `grads` mirrors it. This keeps
//! the optimizer, checkpointing and gradient clipping trivial.

use crate::ops::{add_bias_rows, mm_nn, mm_nt, mm_tn, softmax_row};
use crate::pool::{par_for, par_for_work, ParMut};
use crate::rng::Rng;

const LN_EPS: f32 = 1e-5;

#[derive(Clone, Debug)]
pub struct GptConfig {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub dropout: f32,
    pub bias: bool,
}

/// (offset, length) range into the flat parameter/gradient buffers.
#[derive(Clone, Copy)]
struct R {
    off: usize,
    len: usize,
}

#[derive(Clone, Copy)]
struct LayerR {
    g: R,
    b: Option<R>,
}

#[derive(Clone)]
struct BlockR {
    ln1: LayerR,
    attn_w: R,
    attn_b: Option<R>,
    proj_w: R,
    proj_b: Option<R>,
    ln2: LayerR,
    fc_w: R,
    fc_b: Option<R>,
    mlp_w: R,
    mlp_b: Option<R>,
}

/// Which activation buffer currently holds the latest block output.
#[derive(Clone, Copy, PartialEq)]
enum BufId {
    Emb,
    Ping,
    Pong,
}

enum Fill {
    Normal(f32),
    Ones,
    Zeros,
}

struct ParamBuilder {
    params: Vec<f32>,
    decay: Vec<bool>,
}

impl ParamBuilder {
    fn add(&mut self, len: usize, weight_decay: bool, fill: Fill, rng: &mut Rng) -> R {
        let off = self.params.len();
        match fill {
            Fill::Normal(std) => {
                for _ in 0..len {
                    self.params.push(rng.normal_f32(0.0, std));
                }
            }
            Fill::Ones => self.params.extend(std::iter::repeat(1.0).take(len)),
            Fill::Zeros => self.params.extend(std::iter::repeat(0.0).take(len)),
        }
        self.decay.extend(std::iter::repeat(weight_decay).take(len));
        R { off, len }
    }
}

pub struct Gpt {
    pub cfg: GptConfig,
    pub params: Vec<f32>,
    pub grads: Vec<f32>,
    /// per-element flag: apply AdamW weight decay (all 2-D weights, mirroring
    /// `configure_optimizers` which decays everything with dim >= 2)
    pub decay: Vec<bool>,
    wte: R,
    wpe: R,
    blocks: Vec<BlockR>,
    lnf: LayerR,
}

impl Gpt {
    pub fn new(cfg: GptConfig, rng: &mut Rng) -> Self {
        assert!(cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head");
        let c = cfg.n_embd;
        let v = cfg.vocab_size;
        let t = cfg.block_size;
        let resid_std = 0.02 / (2.0 * cfg.n_layer as f32).sqrt();

        let mut pb = ParamBuilder { params: Vec::new(), decay: Vec::new() };
        let wte = pb.add(v * c, true, Fill::Normal(0.02), rng); // token embeddings, also lm_head (weight tying)
        let wpe = pb.add(t * c, true, Fill::Normal(0.02), rng); // position embeddings
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let ln1g = pb.add(c, false, Fill::Ones, rng);
            let ln1b = cfg.bias.then(|| pb.add(c, false, Fill::Zeros, rng));
            let attn_w = pb.add(c * 3 * c, true, Fill::Normal(0.02), rng);
            let attn_b = cfg.bias.then(|| pb.add(3 * c, false, Fill::Zeros, rng));
            let proj_w = pb.add(c * c, true, Fill::Normal(resid_std), rng);
            let proj_b = cfg.bias.then(|| pb.add(c, false, Fill::Zeros, rng));
            let ln2g = pb.add(c, false, Fill::Ones, rng);
            let ln2b = cfg.bias.then(|| pb.add(c, false, Fill::Zeros, rng));
            let fc_w = pb.add(c * 4 * c, true, Fill::Normal(0.02), rng);
            let fc_b = cfg.bias.then(|| pb.add(4 * c, false, Fill::Zeros, rng));
            let mlp_w = pb.add(4 * c * c, true, Fill::Normal(resid_std), rng);
            let mlp_b = cfg.bias.then(|| pb.add(c, false, Fill::Zeros, rng));
            blocks.push(BlockR {
                ln1: LayerR { g: ln1g, b: ln1b },
                attn_w,
                attn_b,
                proj_w,
                proj_b,
                ln2: LayerR { g: ln2g, b: ln2b },
                fc_w,
                fc_b,
                mlp_w,
                mlp_b,
            });
        }
        let lnfg = pb.add(c, false, Fill::Ones, rng);
        let lnfb = cfg.bias.then(|| pb.add(c, false, Fill::Zeros, rng));
        let lnf = LayerR { g: lnfg, b: lnfb };

        let grads = vec![0.0; pb.params.len()];
        Gpt { cfg, params: pb.params, grads, decay: pb.decay, wte, wpe, blocks, lnf }
    }

    /// Parameter count; by default the position embeddings are subtracted,
    /// like `get_num_params` in model.py.
    pub fn num_params(&self) -> usize {
        self.params.len() - self.wpe.len
    }

    pub fn zero_grad(&mut self) {
        for g in self.grads.iter_mut() {
            *g = 0.0;
        }
    }

    /// Forward pass. Returns the mean cross-entropy loss when `targets` is
    /// given; fills `cache.logits` with the full [B*T, V] logits.
    pub fn forward(
        &self,
        tokens: &[u32],
        targets: Option<&[u32]>,
        training: bool,
        rng: &mut Rng,
        cache: &mut Cache,
        threads: usize,
    ) -> f32 {
        let cfg = self.cfg.clone();
        let t = cfg.block_size.min(tokens.len());
        assert!(tokens.len() % t == 0, "tokens length must be a multiple of block size");
        assert!(t <= cfg.block_size, "sequence longer than block_size");
        let b = tokens.len() / t;
        let m = b * t;
        let c = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = c / nh;
        let use_dropout = training && cfg.dropout > 0.0;
        let drop_scale = if use_dropout { 1.0 / (1.0 - cfg.dropout) } else { 1.0 };

        cache.setup(m, t, &cfg);
        cache.drop_scale = drop_scale;
        cache.tokens.copy_from_slice(tokens);

        let params = self.params.as_slice();
        let Cache {
            tokens: _ct, emb, emb_mask, ping, pong, blocks, lnf_x, lnf_mean, lnf_rstd,
            lnf_out, logits, probs: _probs, ..
        } = cache;

        // token embeddings + position embeddings
        let wte_s = &params[self.wte.off..self.wte.off + self.wte.len];
        let wpe_s = &params[self.wpe.off..self.wpe.off + self.wpe.len];
        let embp = ParMut::new(emb);
        par_for_work(threads, m, 2 * c, |i| {
            let tok = tokens[i] as usize;
            let pos = i % t;
            // SAFETY: each job owns exactly one row of emb.
            let er = unsafe { embp.slice(i * c, (i + 1) * c) };
            for d in 0..c {
                er[d] = wte_s[tok * c + d] + wpe_s[pos * c + d];
            }
        });
        if use_dropout {
            dropout_forward(emb, emb_mask, cfg.dropout, drop_scale, rng);
        } else {
            emb_mask.clear();
        }

        // transformer blocks (ping-pong buffers; source of block 0 is `emb`)
        let mut prev = BufId::Emb;
        for l in 0..cfg.n_layer {
            let dst_id = if l % 2 == 0 { BufId::Ping } else { BufId::Pong };
            let bo = self.blocks[l].clone();
            let args = (b, t, c, nh, hs, use_dropout, drop_scale);
            match (prev, dst_id) {
                (BufId::Emb, BufId::Ping) => block_forward(params, &bo, emb.as_slice(), &mut blocks[l], ping, args, rng, threads),
                (BufId::Emb, BufId::Pong) => block_forward(params, &bo, emb.as_slice(), &mut blocks[l], pong, args, rng, threads),
                (BufId::Ping, BufId::Pong) => block_forward(params, &bo, ping.as_slice(), &mut blocks[l], pong, args, rng, threads),
                (BufId::Pong, BufId::Ping) => block_forward(params, &bo, pong.as_slice(), &mut blocks[l], ping, args, rng, threads),
                _ => unreachable!(),
            }
            prev = dst_id;
        }
        // final activation source
        let last: &[f32] = match prev {
            BufId::Emb => emb.as_slice(),
            BufId::Ping => ping.as_slice(),
            BufId::Pong => pong.as_slice(),
        };

        lnf_x.copy_from_slice(last);
        let lnf_g = &params[self.lnf.g.off..self.lnf.g.off + self.lnf.g.len];
        let lnf_b = self.lnf.b.map(|r| &params[r.off..r.off + r.len]);
        ln_forward(lnf_x, lnf_g, lnf_b, lnf_mean, lnf_rstd, lnf_out, m, c, threads);

        // lm_head shares the token embedding weights (weight tying)
        for x in logits.iter_mut() {
            *x = 0.0;
        }
        mm_nt(logits, lnf_out, wte_s, m, c, v, threads);

        let mut loss = 0.0;
        if let Some(tg) = targets {
            assert_eq!(tg.len(), m);
            let inv_m = 1.0 / m as f32;
            for i in 0..m {
                softmax_row(&mut logits[i * v..(i + 1) * v]);
            }
            probs_from_logits(logits, _probs, m, v);
            for i in 0..m {
                let p = _probs[i * v + tg[i] as usize].max(1e-12);
                loss -= p.ln();
            }
            loss *= inv_m;
        }
        loss
    }

    /// Backward pass: fills `self.grads` (which must have been zeroed).
    pub fn backward(&mut self, cache: &mut Cache, targets: &[u32], threads: usize) {
        let cfg = self.cfg.clone();
        let t = cache.t;
        let b = cache.b;
        let m = b * t;
        let c = cfg.n_embd;
        let v = cfg.vocab_size;
        let nh = cfg.n_head;
        let hs = c / nh;
        let drop_scale = cache.drop_scale;

        let params = self.params.as_slice();
        let grads = self.grads.as_mut_slice();
        let wte = self.wte;
        let wpe = self.wpe;
        let lnf = self.lnf;
        let block_rs = self.blocks.clone();

        let Cache {
            tokens, emb_mask, blocks, lnf_x, lnf_mean, lnf_rstd, lnf_out,
            probs, dx, dx2, dx3, dqkv, d4c, ..
        } = cache;

        let wte_s = &params[wte.off..wte.off + wte.len];

        // dloss/dlogits = (softmax - onehot) / m, reusing the probs buffer
        let dlogits = probs; // &mut Vec<f32>
        let inv_m = 1.0 / m as f32;
        for i in 0..m {
            let row = &mut dlogits[i * v..(i + 1) * v];
            for x in row.iter_mut() {
                *x *= inv_m;
            }
            row[targets[i] as usize] -= inv_m;
        }

        // lm_head backward (weight-tied with wte)
        for x in dx.iter_mut() {
            *x = 0.0;
        }
        mm_nn(dx, dlogits, wte_s, m, v, c, threads); // dx = dlogits @ wte
        mm_tn(
            &mut grads[wte.off..wte.off + wte.len],
            dlogits, lnf_out, m, v, c, threads,
        ); // gwte += dlogits^T @ lnf_out

        // final layernorm backward (in-place on dx)
        let lnf_g = &params[lnf.g.off..lnf.g.off + lnf.g.len];
        let lnf_x_s = lnf_x.as_slice();
        let lnf_mean_s = lnf_mean.as_slice();
        let lnf_rstd_s = lnf_rstd.as_slice();
        let (lnf_dg, lnf_rest) = grads[lnf.g.off..].split_at_mut(c);
        let lnf_db = lnf.b.map(|_| &mut lnf_rest[..c]);
        ln_backward(dx, lnf_x_s, lnf_g, lnf_mean_s, lnf_rstd_s, lnf_dg, lnf_db, m, c, threads);

        // blocks in reverse
        for l in (0..cfg.n_layer).rev() {
            let bo = block_rs[l].clone();
            let bc = &blocks[l];
            block_backward(
                params, grads, &bo, bc,
                dx, dx2, dx3, dqkv, d4c,
                b, t, c, nh, hs, drop_scale, threads,
            );
        }

        // embedding backward (serial: repeated token ids would race otherwise)
        let (g_wte, g_rest) = grads[wte.off..].split_at_mut(wte.len);
        let g_wpe = &mut g_rest[..wpe.len];
        let use_dropout = !emb_mask.is_empty();
        for i in 0..m {
            let tok = tokens[i] as usize;
            let pos = i % t;
            let s = if use_dropout { emb_mask[i] as f32 * drop_scale } else { 1.0 };
            let dr = &dx[i * c..(i + 1) * c];
            for d in 0..c {
                let g = dr[d] * s;
                g_wte[tok * c + d] += g;
                g_wpe[pos * c + d] += g;
            }
        }
    }

    /// Autoregressive generation mirroring `GPT.generate` in model.py.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        rng: &mut Rng,
        threads: usize,
    ) -> Vec<u32> {
        let mut idx: Vec<u32> = prompt.to_vec();
        let mut cache = Cache::default();
        let v = self.cfg.vocab_size;
        let mut row = vec![0.0f32; v];
        for _ in 0..max_new_tokens {
            let start = idx.len().saturating_sub(self.cfg.block_size);
            let ctx = &idx[start..];
            self.forward(ctx, None, false, rng, &mut cache, threads);
            let tt = cache.t;
            row.copy_from_slice(&cache.logits[(tt - 1) * v..tt * v]);
            for x in row.iter_mut() {
                *x /= temperature;
            }
            if top_k > 0 {
                let mut sorted = row.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let thresh = sorted[top_k.min(v) - 1];
                for x in row.iter_mut() {
                    if *x < thresh {
                        *x = f32::NEG_INFINITY;
                    }
                }
            }
            softmax_row(&mut row);
            // multinomial draw
            let u = rng.uniform_f32();
            let mut cum = 0.0f32;
            let mut next = v - 1;
            for (i, &p) in row.iter().enumerate() {
                cum += p;
                if u < cum {
                    next = i;
                    break;
                }
            }
            idx.push(next as u32);
        }
        idx
    }
}

fn probs_from_logits(logits: &[f32], probs: &mut Vec<f32>, m: usize, v: usize) {
    probs.resize(m * v, 0.0);
    probs.copy_from_slice(logits);
}

// ---------------------------------------------------------------------------
// dropout
// ---------------------------------------------------------------------------

fn dropout_forward(x: &mut [f32], mask: &mut Vec<u8>, p: f32, scale: f32, rng: &mut Rng) {
    mask.resize(x.len(), 0);
    for (i, xv) in x.iter_mut().enumerate() {
        let keep = rng.uniform_f32() >= p;
        mask[i] = keep as u8;
        *xv = if keep { *xv * scale } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// layernorm
// ---------------------------------------------------------------------------

fn ln_forward(
    x: &[f32], g: &[f32], bias: Option<&[f32]>,
    mean: &mut [f32], rstd: &mut [f32], y: &mut [f32],
    rows: usize, c: usize, threads: usize,
) {
    let yp = ParMut::new(y);
    let mp = ParMut::new(mean);
    let rp = ParMut::new(rstd);
    par_for_work(threads, rows, 4 * c, |r| {
        let xr = &x[r * c..(r + 1) * c];
        // SAFETY: each job owns one disjoint row of y plus mean[r]/rstd[r].
        let yr = unsafe { yp.slice(r * c, (r + 1) * c) };
        let mut mu = 0.0f32;
        for &v in xr {
            mu += v;
        }
        mu /= c as f32;
        let mut var = 0.0f32;
        for &v in xr {
            let d = v - mu;
            var += d * d;
        }
        var /= c as f32;
        let rs = 1.0 / (var + LN_EPS).sqrt();
        unsafe {
            mp.set(r, mu);
            rp.set(r, rs);
        }
        match bias {
            Some(b) => {
                for d in 0..c {
                    yr[d] = (xr[d] - mu) * rs * g[d] + b[d];
                }
            }
            None => {
                for d in 0..c {
                    yr[d] = (xr[d] - mu) * rs * g[d];
                }
            }
        }
    });
}

/// In-place: `dy` is replaced by dx. Also accumulates dg / db.
fn ln_backward(
    dy: &mut [f32], x: &[f32], g: &[f32], mean: &[f32], rstd: &[f32],
    dg: &mut [f32], db: Option<&mut [f32]>,
    rows: usize, c: usize, threads: usize,
) {
    let dyp = ParMut::new(dy);
    par_for_work(threads, rows, 6 * c, |r| {
        // SAFETY: each job owns one disjoint row of dy.
        let dyr = unsafe { dyp.slice(r * c, (r + 1) * c) };
        let xr = &x[r * c..(r + 1) * c];
        let rs = rstd[r];
        let mu = mean[r];
        let mut s1 = 0.0f32; // sum dy*g
        let mut s2 = 0.0f32; // sum dy*g*xhat
        for d in 0..c {
            let dxh = dyr[d] * g[d];
            let xhat = (xr[d] - mu) * rs;
            s1 += dxh;
            s2 += dxh * xhat;
        }
        let m1 = s1 / c as f32;
        let m2 = s2 / c as f32;
        for d in 0..c {
            let dxh = dyr[d] * g[d];
            let xhat = (xr[d] - mu) * rs;
            dyr[d] = rs * (dxh - m1 - xhat * m2);
        }
    });
    // dg / db reductions, parallel over columns
    let dgp = ParMut::new(dg);
    let dbp = db.map(ParMut::new);
    par_for_work(threads, c, 4 * rows, |d| {
        let mut sg = 0.0f32;
        let mut sb = 0.0f32;
        for r in 0..rows {
            let dyv = dy[r * c + d];
            let xhat = (x[r * c + d] - mean[r]) * rstd[r];
            sg += dyv * xhat;
            sb += dyv;
        }
        // SAFETY: job owns column d of dg/db exclusively.
        unsafe {
            dgp.add_assign(d, sg);
            if let Some(dbp) = dbp {
                dbp.add_assign(d, sb);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// GELU (tanh approximation, as used by GPT-2)
// ---------------------------------------------------------------------------

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF: f32 = 0.044715;

fn gelu_forward(x: &[f32], y: &mut [f32], threads: usize) {
    let n = x.len();
    let yp = ParMut::new(y);
    par_for(threads, n.div_ceil(4096), |chunk| {
        for i in (chunk * 4096)..((chunk + 1) * 4096).min(n) {
            let xv = x[i];
            let u = SQRT_2_OVER_PI * (xv + GELU_COEF * xv * xv * xv);
            // SAFETY: chunks are disjoint.
            unsafe { yp.set(i, 0.5 * xv * (1.0 + u.tanh())) };
        }
    });
}

/// In-place on `dy`; `x` is the pre-activation.
fn gelu_backward(dy: &mut [f32], x: &[f32], threads: usize) {
    let n = x.len();
    let dyp = ParMut::new(dy);
    par_for(threads, n.div_ceil(4096), |chunk| {
        for i in (chunk * 4096)..((chunk + 1) * 4096).min(n) {
            let xv = x[i];
            let u = SQRT_2_OVER_PI * (xv + GELU_COEF * xv * xv * xv);
            let th = u.tanh();
            let d = 0.5 * (1.0 + th)
                + 0.5 * xv * (1.0 - th * th) * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEF * xv * xv);
            // SAFETY: chunks are disjoint.
            unsafe { dyp.set(i, dyp.get(i) * d) };
        }
    });
}

// ---------------------------------------------------------------------------
// transformer block fwd/bwd
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn block_forward(
    params: &[f32], bo: &BlockR, src: &[f32], bc: &mut BlockCache, dst: &mut [f32],
    args: (usize, usize, usize, usize, usize, bool, f32), rng: &mut Rng, threads: usize,
) {
    let (b, t, c, nh, hs, use_dropout, drop_scale) = args;
    let m = b * t;
    let p_drop = if use_dropout { 1.0 - 1.0 / drop_scale } else { 0.0 };
    let BlockCache {
        x_in, ln1_mean, ln1_rstd, ln1_out, qkv, att, att_mask, attn_out, resid_mask, x2,
        ln2_mean, ln2_rstd, ln2_out, fc_out, gelu_out, mlp_out, mlp_mask,
    } = bc;

    x_in.copy_from_slice(src);

    // ln_1
    let ln1_g = &params[bo.ln1.g.off..bo.ln1.g.off + bo.ln1.g.len];
    let ln1_b = bo.ln1.b.map(|r| &params[r.off..r.off + r.len]);
    ln_forward(x_in, ln1_g, ln1_b, ln1_mean, ln1_rstd, ln1_out, m, c, threads);

    // qkv projection
    let attn_w = &params[bo.attn_w.off..bo.attn_w.off + bo.attn_w.len];
    for x in qkv.iter_mut() {
        *x = 0.0;
    }
    mm_nn(qkv, ln1_out, attn_w, m, c, 3 * c, threads);
    if let Some(rb) = bo.attn_b {
        add_bias_rows(qkv, &params[rb.off..rb.off + rb.len], m, 3 * c, threads);
    }

    // attention scores + softmax (causal)
    attention_forward(qkv, att, b, t, c, nh, hs, threads);
    if use_dropout {
        dropout_forward(att, att_mask, p_drop, drop_scale, rng);
    } else {
        att_mask.clear();
    }

    // att @ v
    attention_apply(att, att_mask, drop_scale, qkv, attn_out, b, t, c, nh, hs, threads);

    // output projection + residual dropout + residual add
    let proj_w = &params[bo.proj_w.off..bo.proj_w.off + bo.proj_w.len];
    for x in x2.iter_mut() {
        *x = 0.0;
    }
    mm_nn(x2, attn_out, proj_w, m, c, c, threads);
    if let Some(rb) = bo.proj_b {
        add_bias_rows(x2, &params[rb.off..rb.off + rb.len], m, c, threads);
    }
    if use_dropout {
        dropout_forward(x2, resid_mask, p_drop, drop_scale, rng);
    } else {
        resid_mask.clear();
    }
    for i in 0..m * c {
        x2[i] += x_in[i];
    }

    // ln_2
    let ln2_g = &params[bo.ln2.g.off..bo.ln2.g.off + bo.ln2.g.len];
    let ln2_b = bo.ln2.b.map(|r| &params[r.off..r.off + r.len]);
    ln_forward(x2, ln2_g, ln2_b, ln2_mean, ln2_rstd, ln2_out, m, c, threads);

    // MLP: fc -> gelu -> proj -> dropout -> residual
    let fc_w = &params[bo.fc_w.off..bo.fc_w.off + bo.fc_w.len];
    for x in fc_out.iter_mut() {
        *x = 0.0;
    }
    mm_nn(fc_out, ln2_out, fc_w, m, c, 4 * c, threads);
    if let Some(rb) = bo.fc_b {
        add_bias_rows(fc_out, &params[rb.off..rb.off + rb.len], m, 4 * c, threads);
    }
    gelu_forward(fc_out, gelu_out, threads);
    let mlp_w = &params[bo.mlp_w.off..bo.mlp_w.off + bo.mlp_w.len];
    for x in mlp_out.iter_mut() {
        *x = 0.0;
    }
    mm_nn(mlp_out, gelu_out, mlp_w, m, 4 * c, c, threads);
    if let Some(rb) = bo.mlp_b {
        add_bias_rows(mlp_out, &params[rb.off..rb.off + rb.len], m, c, threads);
    }
    if use_dropout {
        dropout_forward(mlp_out, mlp_mask, p_drop, drop_scale, rng);
    } else {
        mlp_mask.clear();
    }
    for i in 0..m * c {
        dst[i] = x2[i] + mlp_out[i];
    }
}

/// att[b,h,i,j] = softmax_j<=i (q_i . k_j / sqrt(hs))
fn attention_forward(qkv: &[f32], att: &mut [f32], b: usize, t: usize, c: usize, nh: usize, hs: usize, threads: usize) {
    let inv_sqrt_hs = 1.0 / (hs as f32).sqrt();
    let attp = ParMut::new(att);
    par_for(threads, b * nh, |job| {
        let bb = job / nh;
        let h = job % nh;
        let att_base = job * t * t;
        for i in 0..t {
            let qi = (bb * t + i) * 3 * c + h * hs;
            // SAFETY: each (b,h) job owns its disjoint T*T tile of att.
            let row = unsafe { attp.slice(att_base + i * t, att_base + i * t + t) };
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let kj = (bb * t + j) * 3 * c + c + h * hs;
                let mut s = 0.0f32;
                for d in 0..hs {
                    s += qkv[qi + d] * qkv[kj + d];
                }
                let sv = s * inv_sqrt_hs;
                row[j] = sv;
                if sv > max {
                    max = sv;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                let e = (row[j] - max).exp();
                row[j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for j in 0..=i {
                row[j] *= inv;
            }
        }
    });
}

/// y[b,i,h*hs+d] = sum_j att_eff[b,h,i,j] * v[b,j,h*hs+d]
fn attention_apply(
    att: &[f32], att_mask: &[u8], drop_scale: f32, qkv: &[f32], y: &mut [f32],
    b: usize, t: usize, c: usize, nh: usize, hs: usize, threads: usize,
) {
    let masked = !att_mask.is_empty();
    let yp = ParMut::new(y);
    par_for(threads, b * nh, |job| {
        let bb = job / nh;
        let h = job % nh;
        let att_base = job * t * t;
        for i in 0..t {
            let yi = (bb * t + i) * c + h * hs;
            for d in 0..hs {
                // SAFETY: job (bb,h) owns columns h*hs..h*hs+hs of its rows.
                unsafe { yp.set(yi + d, 0.0) };
            }
            for j in 0..=i {
                let mut a = att[att_base + i * t + j];
                if masked {
                    a *= att_mask[att_base + i * t + j] as f32 * drop_scale;
                }
                if a == 0.0 {
                    continue;
                }
                let vj = (bb * t + j) * 3 * c + 2 * c + h * hs;
                for d in 0..hs {
                    unsafe { yp.add_assign(yi + d, a * qkv[vj + d]) };
                }
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn block_backward(
    params: &[f32], grads: &mut [f32], bo: &BlockR, bc: &BlockCache,
    dx: &mut [f32], dx2: &mut [f32], dx3: &mut [f32], dqkv: &mut [f32], d4c: &mut [f32],
    b: usize, t: usize, c: usize, nh: usize, hs: usize, drop_scale: f32, threads: usize,
) {
    let m = b * t;
    let masked_mlp = !bc.mlp_mask.is_empty();
    let masked_resid = !bc.resid_mask.is_empty();

    // ---- MLP branch -------------------------------------------------------
    // d(mlp_out) = grad-of-block-output through the dropout
    let g3: &[f32] = dx;
    let d4cp = ParMut::new(d4c);
    par_for(threads, m * c / 4096 + 1, |chunk| {
        for i in (chunk * 4096)..((chunk + 1) * 4096).min(m * c) {
            let s = if masked_mlp { bc.mlp_mask[i] as f32 * drop_scale } else { 1.0 };
            // SAFETY: chunks are disjoint.
            unsafe { d4cp.set(i, g3[i] * s) }; // d4c reused first as d-mlp-out ([m*c])
        }
    });
    // dgelu = d_mlp_out @ Wmlp^T
    let mlp_w = &params[bo.mlp_w.off..bo.mlp_w.off + bo.mlp_w.len];
    for x in d4c[m * c..m * c + m * 4 * c].iter_mut() {
        *x = 0.0;
    }
    {
        let (d_mlp_out, dgelu) = d4c.split_at_mut(m * c);
        mm_nt(dgelu, d_mlp_out, mlp_w, m, c, 4 * c, threads);
    }
    gelu_backward(&mut d4c[m * c..], &bc.fc_out, threads);
    let df = &d4c[m * c..];

    // grads of MLP weights
    mm_tn(
        &mut grads[bo.mlp_w.off..bo.mlp_w.off + bo.mlp_w.len],
        &bc.gelu_out, &d4c[..m * c], m, 4 * c, c, threads,
    );
    if let Some(rb) = bo.mlp_b {
        bias_grad(&mut grads[rb.off..rb.off + rb.len], &d4c[..m * c], m, c);
    }
    // dh2 = df @ Wfc^T
    let fc_w = &params[bo.fc_w.off..bo.fc_w.off + bo.fc_w.len];
    for x in dx3.iter_mut() {
        *x = 0.0;
    }
    mm_nt(dx3, df, fc_w, m, 4 * c, c, threads);
    mm_tn(
        &mut grads[bo.fc_w.off..bo.fc_w.off + bo.fc_w.len],
        &bc.ln2_out, df, m, c, 4 * c, threads,
    );
    if let Some(rb) = bo.fc_b {
        bias_grad(&mut grads[rb.off..rb.off + rb.len], df, m, 4 * c);
    }
    // ln2 backward + residual connection -> grad wrt x2
    let ln2_g = &params[bo.ln2.g.off..bo.ln2.g.off + bo.ln2.g.len];
    let (ln2_dg, ln2_rest) = grads[bo.ln2.g.off..].split_at_mut(c);
    let ln2_db = bo.ln2.b.map(|_| &mut ln2_rest[..c]);
    ln_backward(dx3, &bc.x2, ln2_g, &bc.ln2_mean, &bc.ln2_rstd,
        ln2_dg, ln2_db, m, c, threads);
    let g3: &[f32] = dx;
    for i in 0..m * c {
        dx2[i] = dx3[i] + g3[i];
    }

    // ---- attention branch ---------------------------------------------------
    // do = grad wrt proj output (through resid dropout)
    let dx3p = ParMut::new(dx3);
    par_for(threads, m * c / 4096 + 1, |chunk| {
        for i in (chunk * 4096)..((chunk + 1) * 4096).min(m * c) {
            let s = if masked_resid { bc.resid_mask[i] as f32 * drop_scale } else { 1.0 };
            // SAFETY: chunks are disjoint.
            unsafe { dx3p.set(i, dx2[i] * s) };
        }
    });
    // dy_attn = do @ Wproj^T
    let proj_w = &params[bo.proj_w.off..bo.proj_w.off + bo.proj_w.len];
    let mut dy_attn = vec![0.0f32; m * c];
    mm_nt(&mut dy_attn, &dx3[..m * c], proj_w, m, c, c, threads);
    mm_tn(
        &mut grads[bo.proj_w.off..bo.proj_w.off + bo.proj_w.len],
        &bc.attn_out, &dx3[..m * c], m, c, c, threads,
    );
    if let Some(rb) = bo.proj_b {
        bias_grad(&mut grads[rb.off..rb.off + rb.len], &dx3[..m * c], m, c);
    }
    // attention backward -> dqkv
    for x in dqkv.iter_mut() {
        *x = 0.0;
    }
    attention_backward(&dy_attn, &bc.qkv, &bc.att, &bc.att_mask, drop_scale, dqkv,
        b, t, c, nh, hs, threads);
    // dh = dqkv @ Wattn^T
    let attn_w = &params[bo.attn_w.off..bo.attn_w.off + bo.attn_w.len];
    for x in dx3.iter_mut() {
        *x = 0.0;
    }
    mm_nt(dx3, dqkv, attn_w, m, 3 * c, c, threads);
    mm_tn(
        &mut grads[bo.attn_w.off..bo.attn_w.off + bo.attn_w.len],
        &bc.ln1_out, dqkv, m, c, 3 * c, threads,
    );
    if let Some(rb) = bo.attn_b {
        bias_grad(&mut grads[rb.off..rb.off + rb.len], dqkv, m, 3 * c);
    }
    // ln1 backward + residual -> grad wrt block input
    let ln1_g = &params[bo.ln1.g.off..bo.ln1.g.off + bo.ln1.g.len];
    let (ln1_dg, ln1_rest) = grads[bo.ln1.g.off..].split_at_mut(c);
    let ln1_db = bo.ln1.b.map(|_| &mut ln1_rest[..c]);
    ln_backward(dx3, &bc.x_in, ln1_g, &bc.ln1_mean, &bc.ln1_rstd,
        ln1_dg, ln1_db, m, c, threads);
    for i in 0..m * c {
        dx[i] = dx3[i] + dx2[i];
    }
}

fn bias_grad(db: &mut [f32], dy: &[f32], rows: usize, cols: usize) {
    for o in 0..cols {
        let mut s = 0.0f32;
        for m in 0..rows {
            s += dy[m * cols + o];
        }
        db[o] += s;
    }
}

/// Computes dqkv from dy_attn. `att` holds the softmax probabilities
/// (pre-dropout); `att_mask` the dropout mask applied to them.
#[allow(clippy::too_many_arguments)]
fn attention_backward(
    dy: &[f32], qkv: &[f32], att: &[f32], att_mask: &[u8], drop_scale: f32,
    dqkv: &mut [f32], b: usize, t: usize, c: usize, nh: usize, hs: usize, threads: usize,
) {
    let masked = !att_mask.is_empty();
    let inv_sqrt_hs = 1.0 / (hs as f32).sqrt();
    let dqp = ParMut::new(dqkv);
    par_for(threads, b * nh, |job| {
        let bb = job / nh;
        let h = job % nh;
        let att_base = job * t * t;
        let mut u_row = vec![0.0f32; t];
        for i in 0..t {
            let row_i = bb * t + i;
            let dy_i = row_i * c + h * hs;
            // u_j = (dy_i . v_j) * mask * scale   (grad wrt softmax output)
            let mut rowdot = 0.0f32;
            for j in 0..=i {
                let vj = (bb * t + j) * 3 * c + 2 * c + h * hs;
                let mut s = 0.0f32;
                for d in 0..hs {
                    s += dy[dy_i + d] * qkv[vj + d];
                }
                if masked {
                    s *= att_mask[att_base + i * t + j] as f32 * drop_scale;
                }
                u_row[j] = s;
                rowdot += att[att_base + i * t + j] * s;
            }
            // dv_j += att_eff[i,j] * dy_i ; dq_i / dk_j via dqk
            let dq_i = row_i * 3 * c + h * hs;
            for j in 0..=i {
                let a = att[att_base + i * t + j];
                let dqk = a * (u_row[j] - rowdot) * inv_sqrt_hs;
                if dqk != 0.0 {
                    let q_j = (bb * t + j) * 3 * c + h * hs;
                    let k_j = (bb * t + j) * 3 * c + c + h * hs;
                    for d in 0..hs {
                        // SAFETY: job (bb,h) owns the q/k/v columns of head h
                        // within its own batch rows, exclusively.
                        unsafe {
                            dqp.add_assign(dq_i + d, dqk * qkv[k_j + d]);
                            dqp.add_assign(k_j + d, dqk * qkv[q_j + d]);
                        }
                    }
                }
                let a_eff = if masked { a * att_mask[att_base + i * t + j] as f32 * drop_scale } else { a };
                if a_eff != 0.0 {
                    let v_j = (bb * t + j) * 3 * c + 2 * c + h * hs;
                    for d in 0..hs {
                        unsafe { dqp.add_assign(v_j + d, a_eff * dy[dy_i + d]) };
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// forward/backward activation cache
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct BlockCache {
    x_in: Vec<f32>,
    ln1_mean: Vec<f32>,
    ln1_rstd: Vec<f32>,
    ln1_out: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    att_mask: Vec<u8>,
    attn_out: Vec<f32>,
    resid_mask: Vec<u8>,
    x2: Vec<f32>,
    ln2_mean: Vec<f32>,
    ln2_rstd: Vec<f32>,
    ln2_out: Vec<f32>,
    fc_out: Vec<f32>,
    gelu_out: Vec<f32>,
    mlp_out: Vec<f32>,
    mlp_mask: Vec<u8>,
}

#[derive(Default)]
pub struct Cache {
    pub b: usize,
    pub t: usize,
    drop_scale: f32,
    tokens: Vec<u32>,
    emb: Vec<f32>,
    emb_mask: Vec<u8>,
    ping: Vec<f32>,
    pong: Vec<f32>,
    blocks: Vec<BlockCache>,
    lnf_x: Vec<f32>,
    lnf_mean: Vec<f32>,
    lnf_rstd: Vec<f32>,
    lnf_out: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    // backward scratch
    dx: Vec<f32>,
    dx2: Vec<f32>,
    dx3: Vec<f32>,
    dqkv: Vec<f32>,
    d4c: Vec<f32>,
}

impl Cache {
    fn setup(&mut self, m: usize, t: usize, cfg: &GptConfig) {
        self.b = m / t;
        self.t = t;
        let c = cfg.n_embd;
        let v = cfg.vocab_size;
        self.tokens.resize(m, 0);
        self.emb.resize(m * c, 0.0);
        self.ping.resize(m * c, 0.0);
        self.pong.resize(m * c, 0.0);
        self.lnf_x.resize(m * c, 0.0);
        self.lnf_mean.resize(m, 0.0);
        self.lnf_rstd.resize(m, 0.0);
        self.lnf_out.resize(m * c, 0.0);
        self.logits.resize(m * v, 0.0);
        self.dx.resize(m * c, 0.0);
        self.dx2.resize(m * c, 0.0);
        self.dx3.resize(m * c, 0.0);
        self.dqkv.resize(m * 3 * c, 0.0);
        self.d4c.resize(m * c + m * 4 * c, 0.0);
        while self.blocks.len() < cfg.n_layer {
            self.blocks.push(BlockCache::default());
        }
        let att_len = self.b * cfg.n_head * t * t;
        for bc in self.blocks.iter_mut().take(cfg.n_layer) {
            bc.x_in.resize(m * c, 0.0);
            bc.ln1_mean.resize(m, 0.0);
            bc.ln1_rstd.resize(m, 0.0);
            bc.ln1_out.resize(m * c, 0.0);
            bc.qkv.resize(m * 3 * c, 0.0);
            bc.att.resize(att_len, 0.0);
            bc.attn_out.resize(m * c, 0.0);
            bc.x2.resize(m * c, 0.0);
            bc.ln2_mean.resize(m, 0.0);
            bc.ln2_rstd.resize(m, 0.0);
            bc.ln2_out.resize(m * c, 0.0);
            bc.fc_out.resize(m * 4 * c, 0.0);
            bc.gelu_out.resize(m * 4 * c, 0.0);
            bc.mlp_out.resize(m * c, 0.0);
        }
    }
}
